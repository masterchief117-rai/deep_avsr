"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/smeetrs/deep_avsr
"""

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, shutil

from config import args
from models.video_net import VideoNet
from data.lrs2_dataset import LRS2Pretrain
from data.utils import collate_fn
from utils.general import num_params, train, evaluate

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

def main():


    if torch.cuda.is_available():  
        for i in range(torch.cuda.device_count()):  
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")  
            print(f"  - 显存总量: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")  
            print(f"  - 计算能力: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")  
    else:  
        print("没有可用的 GPU")  
    matplotlib.use("Agg")
    np.random.seed(args["SEED"])
    torch.manual_seed(args["SEED"])
    gpuAvailable = torch.cuda.is_available()
    device = torch.device("cuda" if gpuAvailable else "cpu")
    print(f"当前正在使用的设备是：{device}================")

    # 打印详细的GPU信息  
    if gpuAvailable:  
        print(f"可用 GPU 数量: {torch.cuda.device_count()}")  
        for i in range(torch.cuda.device_count()):  
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")  
        current_device = torch.cuda.current_device()  
        print(f"当前主 GPU 设备编号: {current_device}")  
        print(f"当前主 GPU 设备名称: {torch.cuda.get_device_name(current_device)}")  
        print(f"GPU 显存总量: {torch.cuda.get_device_properties(current_device).total_memory / 1e9:.2f} GB")  
        print(f"GPU 计算能力: {torch.cuda.get_device_properties(current_device).major}.{torch.cuda.get_device_properties(current_device).minor}")  



    kwargs = {"num_workers": args["NUM_WORKERS"], "pin_memory": True} if gpuAvailable else {}
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    #declaring the pretrain and the preval datasets and the corresponding dataloaders
    videoParams = {"videoFPS":args["VIDEO_FPS"]}
    pretrainData = LRS2Pretrain("pretrain", args["DATA_DIRECTORY"], args["PRETRAIN_NUM_WORDS"], args["CHAR_TO_INDEX"], args["STEP_SIZE"],
                                videoParams)
    pretrainLoader = DataLoader(pretrainData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=True, **kwargs)
    prevalData = LRS2Pretrain("preval", args["DATA_DIRECTORY"], args["PRETRAIN_NUM_WORDS"], args["CHAR_TO_INDEX"], args["STEP_SIZE"],
                              videoParams)
    prevalLoader = DataLoader(prevalData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=True, **kwargs)


    #declaring the model, optimizer, scheduler and the loss function
    model = VideoNet(args["TX_NUM_FEATURES"], args["TX_ATTENTION_HEADS"], args["TX_NUM_LAYERS"], args["PE_MAX_LENGTH"],
                     args["TX_FEEDFORWARD_DIM"], args["TX_DROPOUT"], args["NUM_CLASSES"])
    
    if gpuAvailable and torch.cuda.device_count() > 1:  
        print("使用 DataParallel 分布模型到多个 GPU 上")  
        model = nn.DataParallel(model)  
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args["INIT_LR"], betas=(args["MOMENTUM1"], args["MOMENTUM2"]))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args["LR_SCHEDULER_FACTOR"],
                                                     patience=args["LR_SCHEDULER_WAIT"], threshold=args["LR_SCHEDULER_THRESH"],
                                                     threshold_mode="abs", min_lr=args["FINAL_LR"], verbose=True)
    loss_function = nn.CTCLoss(blank=0, zero_infinity=False)


    #removing the checkpoints directory if it exists and remaking it
    if os.path.exists(args["CODE_DIRECTORY"] + "/checkpoints"):
        while True:
            ch = input("Continue and remove the 'checkpoints' directory? y/n: ")
            if ch == "y":
                break
            elif ch == "n":
                exit()
            else:
                print("Invalid input")
        shutil.rmtree(args["CODE_DIRECTORY"] + "/checkpoints")

    os.mkdir(args["CODE_DIRECTORY"] + "/checkpoints")
    os.mkdir(args["CODE_DIRECTORY"] + "/checkpoints/models")
    os.mkdir(args["CODE_DIRECTORY"] + "/checkpoints/plots")


     # 加载预训练权重  
    if args["PRETRAINED_MODEL_FILE"] is not None:  
        print("\n\nPre-trained Model File: %s" % (args["PRETRAINED_MODEL_FILE"]))  
        print("\nLoading the pre-trained model .... \n")  
        pretrained_dict = torch.load(args["CODE_DIRECTORY"] + args["PRETRAINED_MODEL_FILE"], map_location=device)  

        # 检查当前模型是否是 DataParallel  
        if isinstance(model, nn.DataParallel):  
            # 如果是 DataParallel，添加 `module.` 前缀  
            pretrained_dict = {"module." + k: v for k, v in pretrained_dict.items() if "module." + k in model.state_dict()}  
        else:  
            # 如果不是 DataParallel，移除 `module.` 前缀  
            pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items() if k.replace("module.", "") in model.state_dict()}  

        # 加载权重  
        model.load_state_dict(pretrained_dict)  
        print("Loading Done.\n")  


    trainingLossCurve = list()
    validationLossCurve = list()
    trainingWERCurve = list()
    validationWERCurve = list()


    #printing the total and trainable parameters in the model
    numTotalParams, numTrainableParams = num_params(model)
    print("\nNumber of total parameters in the model = %d" %(numTotalParams))
    print("Number of trainable parameters in the model = %d\n" %(numTrainableParams))

    print("Number of Words = %d" %(args["PRETRAIN_NUM_WORDS"]))
    print("\nPretraining the model .... \n")

    trainParams = {"spaceIx":args["CHAR_TO_INDEX"][" "], "eosIx":args["CHAR_TO_INDEX"]["<EOS>"]}
    valParams = {"decodeScheme":"greedy", "spaceIx":args["CHAR_TO_INDEX"][" "], "eosIx":args["CHAR_TO_INDEX"]["<EOS>"]}

    for step in range(args["NUM_STEPS"]):

        #train the model for one step
        trainingLoss, trainingCER, trainingWER = train(model, pretrainLoader, optimizer, loss_function, device, trainParams)
        trainingLossCurve.append(trainingLoss)
        trainingWERCurve.append(trainingWER)

        #evaluate the model on validation set
        validationLoss, validationCER, validationWER = evaluate(model, prevalLoader, loss_function, device, valParams)
        validationLossCurve.append(validationLoss)
        validationWERCurve.append(validationWER)

        #printing the stats after each step
        print("Step: %03d || Tr.Loss: %.6f  Val.Loss: %.6f || Tr.CER: %.3f  Val.CER: %.3f || Tr.WER: %.3f  Val.WER: %.3f"
              %(step, trainingLoss, validationLoss, trainingCER, validationCER, trainingWER, validationWER))

        #make a scheduler step
        scheduler.step(validationWER)


        #saving the model weights and loss/metric curves in the checkpoints directory after every few steps
        if ((step%args["SAVE_FREQUENCY"] == 0) or (step == args["NUM_STEPS"]-1)) and (step != 0):

            savePath = args["CODE_DIRECTORY"] + "/checkpoints/models/pretrain_{:03d}w-step_{:04d}-wer_{:.3f}.pt".format(args["PRETRAIN_NUM_WORDS"],
                                                                                                                        step, validationWER)
            torch.save(model.state_dict(), savePath)

            plt.figure()
            plt.title("Loss Curves")
            plt.xlabel("Step No.")
            plt.ylabel("Loss value")
            plt.plot(list(range(1, len(trainingLossCurve)+1)), trainingLossCurve, "blue", label="Train")
            plt.plot(list(range(1, len(validationLossCurve)+1)), validationLossCurve, "red", label="Validation")
            plt.legend()
            plt.savefig(args["CODE_DIRECTORY"] + "/checkpoints/plots/pretrain_{:03d}w-step_{:04d}-loss.png".format(args["PRETRAIN_NUM_WORDS"], step))
            plt.close()

            plt.figure()
            plt.title("WER Curves")
            plt.xlabel("Step No.")
            plt.ylabel("WER")
            plt.plot(list(range(1, len(trainingWERCurve)+1)), trainingWERCurve, "blue", label="Train")
            plt.plot(list(range(1, len(validationWERCurve)+1)), validationWERCurve, "red", label="Validation")
            plt.legend()
            plt.savefig(args["CODE_DIRECTORY"] + "/checkpoints/plots/pretrain_{:03d}w-step_{:04d}-wer.png".format(args["PRETRAIN_NUM_WORDS"], step))
            plt.close()


    print("\nPretraining Done.\n")

    return

def auto_run_curriculum():  
    """  
    该函数实现“课程学习”模式，每次自动更新 PRETRAIN_NUM_WORDS 并调用 main()。  
    可以自行调整 word_sequence。  
    """  
    word_sequence = [1, 2, 3, 5, 7, 9, 13, 17, 21, 29, 37]  # 11次迭代  
    for w in word_sequence:  
        args["PRETRAIN_NUM_WORDS"] = w  
        print(f"\n========== 即将使用 {w} 个单词进行预训练 ==========\n")  
        main()  



if __name__ == "__main__":
    main()