"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/smeetrs/deep_avsr
"""

import torch
from tqdm import tqdm

from .metrics import compute_cer, compute_wer
from .decoders import ctc_greedy_decode, ctc_search_decode



def num_params(model):
    """
    Function that outputs the number of total and trainable paramters in the model.
    """
    numTotalParams = sum([params.numel() for params in model.parameters()])
    numTrainableParams = sum([params.numel() for params in model.parameters() if params.requires_grad])
    return numTotalParams, numTrainableParams



def train(model, trainLoader, optimizer, loss_function, device, trainParams):

    """
    Function to train the model for one iteration. (Generally, one iteration = one epoch, but here it is one step).
    It also computes the training loss, CER and WER. The CTC decode scheme is always 'greedy' here.
    """

    trainingLoss = 0
    trainingCER = 0
    trainingWER = 0

    for batch, (inputBatch, targetBatch, inputLenBatch, targetLenBatch) in enumerate(tqdm(trainLoader, leave=False, desc="Train",
                                                                                          ncols=75)):

        inputBatch, targetBatch = (inputBatch.float()).to(device), (targetBatch.int()).to(device)
        inputLenBatch, targetLenBatch = (inputLenBatch.int()).to(device), (targetLenBatch.int()).to(device)

        optimizer.zero_grad()
        model.train()
        outputBatch = model(inputBatch)
        with torch.backends.cudnn.flags(enabled=False):
            loss = loss_function(outputBatch, targetBatch, inputLenBatch, targetLenBatch)
        loss.backward()
        optimizer.step()

        trainingLoss = trainingLoss + loss.item()
        predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch.detach(), inputLenBatch, trainParams["eosIx"])
        trainingCER = trainingCER + compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
        trainingWER = trainingWER + compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, trainParams["spaceIx"])

    trainingLoss = trainingLoss/len(trainLoader)
    trainingCER = trainingCER/len(trainLoader)
    trainingWER = trainingWER/len(trainLoader)
    return trainingLoss, trainingCER, trainingWER



def evaluate(model, evalLoader, loss_function, device, evalParams):

    """
    Function to evaluate the model over validation/test set. It computes the loss, CER and WER over the evaluation set.
    The CTC decode scheme can be set to either 'greedy' or 'search'.
    """

    evalLoss = 0
    evalCER = 0
    evalWER = 0

    for batch, (inputBatch, targetBatch, inputLenBatch, targetLenBatch) in enumerate(tqdm(evalLoader, leave=False, desc="Eval",
                                                                                          ncols=75)):

        inputBatch, targetBatch = (inputBatch.float()).to(device), (targetBatch.int()).to(device)
        inputLenBatch, targetLenBatch = (inputLenBatch.int()).to(device), (targetLenBatch.int()).to(device)

        model.eval()
        with torch.no_grad():
            outputBatch = model(inputBatch)
            with torch.backends.cudnn.flags(enabled=False):
                loss = loss_function(outputBatch, targetBatch, inputLenBatch, targetLenBatch)

        evalLoss = evalLoss + loss.item()
        if evalParams["decodeScheme"] == "greedy":
            predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch, inputLenBatch, evalParams["eosIx"])
        elif evalParams["decodeScheme"] == "search":
            predictionBatch, predictionLenBatch = ctc_search_decode(outputBatch, inputLenBatch, evalParams["beamSearchParams"],
                                                                    evalParams["spaceIx"], evalParams["eosIx"], evalParams["lm"])
        else:
            print("Invalid Decode Scheme")
            exit()

        evalCER = evalCER + compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
        evalWER = evalWER + compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, evalParams["spaceIx"])

    evalLoss = evalLoss/len(evalLoader)
    evalCER = evalCER/len(evalLoader)
    evalWER = evalWER/len(evalLoader)
    return evalLoss, evalCER, evalWER

#一个错误处理版本的 train_pretrain 函数
def train_pretrain(model, trainLoader, optimizer, loss_function, device, trainParams):  
    """  
    Function to train the model for one iteration. (Modified to handle empty dataset errors)  
    """  
    
    model.train()  
    epoch_loss = 0  
    epoch_cer = 0  
    epoch_wer = 0  
    
    # 设置最大尝试次数，避免无限循环  
    max_attempts = 5  
    current_attempt = 0  
    batch_count = 0  
    
    # 使用tqdm可视化训练进度  
    try:  
        for batch, (inputBatch, targetBatch, inputLenBatch, targetLenBatch) in enumerate(trainLoader):  
            optimizer.zero_grad()  
            
            inputBatch, targetBatch = inputBatch.to(device), targetBatch.to(device)  
            inputLenBatch, targetLenBatch = inputLenBatch.to(device), targetLenBatch.to(device)  
            
            outputBatch = model(inputBatch)  
            
            with torch.backends.cudnn.flags(enabled=False):  
                loss = loss_function(outputBatch.transpose(0, 1), targetBatch, inputLenBatch, targetLenBatch)  
            
            loss.backward()  
            optimizer.step()  
            
            epoch_loss += loss.item()  
            
            # 计算字符错误率和词错误率  
            predictionBatch, predictionLenBatch = model.decode(outputBatch, inputLenBatch)  
            
            # 累加错误率  
            for i in range(predictionBatch.size(0)):  
                target = targetBatch[i][:targetLenBatch[i]].tolist()  
                targetWords = " ".join([chr(idx) for idx in target]).split(" ")  
                
                prediction = predictionBatch[i][:predictionLenBatch[i]].tolist()  
                predictionWords = " ".join([chr(idx) for idx in prediction]).split(" ")  
                
                # CER计算  
                cerLength = max(len(target), len(prediction))  
                distance = editdistance.eval(target, prediction)  
                epoch_cer += (distance / cerLength)  
                
                # WER计算  
                wer_length = max(len(targetWords), len(predictionWords))  
                distance = editdistance.eval(targetWords, predictionWords)  
                epoch_wer += (distance / wer_length) if wer_length > 0 else 0  
            
            batch_count += 1  
            
            # 如果已经处理了足够的批次，提前返回  
            if batch_count >= 10:  # 限制每个训练步骤的批次数，可以根据需要调整  
                break  
                
    except Exception as e:  
        # 如果是空数据集或其他错误，记录错误并返回上一次成功的指标  
        print(f"Error during training batch: {str(e)}")  
        if batch_count == 0:  
            # 如果一个批次都没成功，返回默认值  
            return 999.0, 1.0, 1.0  
    
    # 计算平均损失和错误率  
    if batch_count > 0:  
        epoch_loss /= batch_count  
        epoch_cer /= batch_count  
        epoch_wer /= batch_count  
    else:  
        # 如果没有成功的批次，返回默认值  
        epoch_loss, epoch_cer, epoch_wer = 999.0, 1.0, 1.0  
        
    return epoch_loss, epoch_cer, epoch_wer  