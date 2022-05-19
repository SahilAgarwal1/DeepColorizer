import time
from tqdm import tqdm
import torch
import torch.nn as nn
from src.data.make_dataset import FaceImagesDataset, train_transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from modules import CNN_Halfing_Block


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 10
epochs = 200
TRAIN_SPLIT = 0.95
VAL_SPIT = 1 - TRAIN_SPLIT


# Get Data and Split into Train and Validation Set
print("[INFO] loading the dataset...")
face_dataset = FaceImagesDataset(root_dir='../../data/face_images_raw', transform=train_transforms, mean_chrominance=1)
print("[INFO] generating the train/validation split...")
train_set, val_set = torch.utils.data.random_split(face_dataset,[int(TRAIN_SPLIT * len(face_dataset)), int(VAL_SPIT * len(face_dataset))])



#Define the model
print("[INFO] Initializing the Model")

RegressorNet = nn.Sequential(
    CNN_Halfing_Block(1, 3),
    CNN_Halfing_Block(3, 3),
    CNN_Halfing_Block(3, 3),
    CNN_Halfing_Block(3, 3),
    CNN_Halfing_Block(3, 3),
    CNN_Halfing_Block(3, 3),
    CNN_Halfing_Block(3, 2)
)



#Model Training
RegressorNet.to(device)
trainDataLoader = DataLoader(train_set, shuffle= True, batch_size= BATCH_SIZE)
ValDataLoader = DataLoader(val_set, shuffle=True, batch_size= BATCH_SIZE)
opt = Adam(RegressorNet.parameters(), lr=0.005)
loss = nn.MSELoss()


print("[INFO] training the network...")
startTime = time.time()


for e in range(0,epochs):
    with tqdm(trainDataLoader, unit="Batch") as tepoch:
        RegressorNet.train()

        totalTrainLoss = 0
        totalValLoss = 0

        for(x,y) in tepoch:

            tepoch.set_description(f"Epoch {e}")
            (x,y) = (x.to(device), y.to(device))

            pred = RegressorNet(x.float())

            l = loss(pred, y)

            opt.zero_grad()

            l.backward()

            opt.step()

            totalTrainLoss += l.item()

        tepoch.set_postfix(l = l.item())


        #calculating validation loss now
        with torch.no_grad():
            #set model in eval mode
            RegressorNet.eval()

            #loop over the validation set
            for (x,y) in ValDataLoader:
                # send the inputs to device
                (x,y) = (x.to(device), y.to(device))

                    # make predictions
                pred = RegressorNet(x)
                totalValLoss += loss(pred, y)


        avgTrainLoss = totalTrainLoss/(len(train_set)//BATCH_SIZE)
        avgValLoss = totalValLoss/(len(val_set)//BATCH_SIZE)

        print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
        print("Train loss: {:.6f}".format(
            avgTrainLoss))
        print("Val loss: {:.6f}\n".format(
            avgValLoss))

endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))






