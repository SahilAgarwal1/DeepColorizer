import time
from tqdm import tqdm
from modules import CNN_Halfing_Block, CNN_Upsampling_Block, CNN_Upsampling_Block_Output
import torch
import torch.nn as nn
from src.data.make_dataset import FaceImagesDataset, train_transforms
from torch.utils.data import DataLoader
from torch.optim import Adam


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 10
epochs = 5
TRAIN_SPLIT = 0.88
VAL_SPIT = 1 - TRAIN_SPLIT
LEARNING_RATE = 0.0001


# Get Data and Split into Train and Validation Set
print("[INFO] loading the dataset...")
face_dataset = FaceImagesDataset(root_dir='../../data/face_images_raw', transform=train_transforms, mean_chrominance=0)
print("[INFO] generating the train/validation split...")
train_set, val_set = torch.utils.data.random_split(face_dataset,[int(TRAIN_SPLIT * len(face_dataset)), int(VAL_SPIT * len(face_dataset))])


#Define the Model

print("[INFO] Initializing the Model")

ColorizerNet = nn.Sequential(
    CNN_Halfing_Block(1, 6),
    CNN_Halfing_Block(6, 12),
    CNN_Halfing_Block(12, 24),
    CNN_Halfing_Block(24, 24),
    CNN_Halfing_Block(24, 48),
    CNN_Upsampling_Block(48,24),
    CNN_Upsampling_Block(24,12),
    CNN_Upsampling_Block(12,12),
    CNN_Upsampling_Block(12,6),
    CNN_Upsampling_Block_Output(6,2)

)




#Model Training
ColorizerNet.to(device)
trainDataLoader = DataLoader(train_set, shuffle= True, batch_size= BATCH_SIZE)
ValDataLoader = DataLoader(val_set, shuffle=True, batch_size= BATCH_SIZE)
opt = Adam(ColorizerNet.parameters(), lr=LEARNING_RATE)
loss = nn.MSELoss()


print("[INFO] training the network...")
startTime = time.time()


for e in range(0,epochs):
    with tqdm(trainDataLoader, unit="Batch") as tepoch:
        ColorizerNet.train()

        totalTrainLoss = 0
        totalValLoss = 0

        for(x,y) in tepoch:


            tepoch.set_description(f"Epoch {e}")
            (x,y) = (x.to(device), y.to(device))

            pred = ColorizerNet(x.float())

            l = loss(pred, y)

            opt.zero_grad()

            l.backward()

            opt.step()

            totalTrainLoss += l.item()

        tepoch.set_postfix(train_loss = l.item())


        #calculating validation loss now
        with torch.no_grad():
            #set model in eval mode
            ColorizerNet.eval()

            #loop over the validation set
            for (x,y) in ValDataLoader:
                # send the inputs to device
                (x,y) = (x.to(device), y.to(device))

                    # make predictions
                pred = ColorizerNet(x)
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

model_scripted = torch.jit.script(ColorizerNet)
model_scripted.save("../../models/ColorizerModel.pt")






