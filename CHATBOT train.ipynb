{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import numpy as np\n",
    "import torch.nn as NN \n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "from nltk_utils import tokenize, stem, bag_of_words\n",
    "\n",
    "with open('intents.json', 'r') as f:\n",
    "    intents = json.load(f)\n",
    "\n",
    "\n",
    "device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Buy', 'Car Brand', 'Car Type', 'Goodbye', 'Greeting', 'Noanswer', 'Purchasing', 'Thanks']\n",
      "[(['Hi'], 'Greeting'), (['How', 'are', 'you', '?'], 'Greeting'), (['Is', 'anymore', 'there', '?'], 'Greeting'), (['Hey'], 'Greeting'), (['Good', 'day'], 'Greeting'), (['I', 'want', 'to', 'buy', 'a', 'car'], 'Buy'), (['I', 'am', 'looking', 'for', 'a', 'car'], 'Buy'), (['I', 'need', 'a', 'car'], 'Buy'), (['car'], 'Buy'), (['Jeep'], 'Buy'), (['New'], 'Car Type'), (['Second', 'hand'], 'Car Type'), (['Branded'], 'Car Type'), (['old'], 'Car Type'), (['x'], 'Car Brand'), (['Y'], 'Car Brand'), (['Z'], 'Car Brand'), (['ABC'], 'Purchasing'), (['XYZ'], 'Purchasing'), (['PQR'], 'Purchasing'), (['A'], 'Purchasing'), (['B'], 'Purchasing'), (['c'], 'Purchasing'), (['XY'], 'Purchasing'), (['YZ'], 'Purchasing'), (['ZA'], 'Purchasing'), (['PQ'], 'Purchasing'), (['QR'], 'Purchasing'), (['RA'], 'Purchasing'), (['Bye'], 'Goodbye'), (['Thanks', 'for', 'visiting', 'our', 'site'], 'Goodbye'), (['See', 'you', 'later'], 'Goodbye'), (['Goodbye'], 'Goodbye'), (['Nice', 'chatting', 'to', 'you', ',', 'bye'], 'Goodbye'), (['Till', 'next', 'time'], 'Goodbye'), (['Thanks'], 'Thanks'), (['Thank', 'you'], 'Thanks'), (['That', \"'s\", 'helpful'], 'Thanks'), (['Awesome', ',', 'thanks'], 'Thanks'), (['Thanks', 'for', 'helping', 'me'], 'Thanks')]\n"
     ]
    }
   ],
   "source": [
    "all_words =[]\n",
    "tags=[]\n",
    "xy=[]\n",
    "for intent in intents['intents']:\n",
    "    tag= intent['tag']\n",
    "    tags.append(tag)\n",
    "    for pattern in intent['patterns']:\n",
    "        w=tokenize(pattern)\n",
    "        all_words.extend(w)\n",
    "        xy.append((w, tag, ))\n",
    "ignore_words = ['?', '!', '.', ',']\n",
    "all_words = [stem(w) for w in all_words if w not in ignore_words]\n",
    "all_words = sorted(set(all_words))\n",
    "tags = sorted(set(tags))\n",
    "print(tags)\n",
    "print(xy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0. 0. 0. ... 0. 0. 0.]\n",
      " [0. 0. 0. ... 0. 0. 0.]\n",
      " [0. 0. 0. ... 0. 0. 0.]\n",
      " ...\n",
      " [1. 0. 0. ... 0. 0. 0.]\n",
      " [0. 0. 0. ... 0. 0. 0.]\n",
      " [0. 0. 0. ... 0. 0. 0.]]\n",
      "[4 4 4 4 4 0 0 0 0 0 2 2 2 2 1 1 1 6 6 6 6 6 6 6 6 6 6 6 6 3 3 3 3 3 3 7 7\n",
      " 7 7 7]\n"
     ]
    }
   ],
   "source": [
    "\n",
    "X_train =[]\n",
    "Y_train =[]\n",
    "for (pattern_sentence, tag) in xy:\n",
    "    bag = bag_of_words(pattern_sentence, all_words)\n",
    "    X_train.append(bag)\n",
    "\n",
    "    label = tags.index(tag)\n",
    "    Y_train.append(label)\n",
    "\n",
    "X_train =np.array(X_train)\n",
    "Y_train=np.array(Y_train)\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "print(X_train)\n",
    "print(Y_train)\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "class ChatDataset(Dataset):\n",
    "    def __init__(self):\n",
    "        self.n_samples= len(X_train)\n",
    "        self.x_data=X_train\n",
    "        self.y_data=Y_train\n",
    "  \n",
    "    def __getitem__(self, index):\n",
    "        return self.x_data[index], self.y_data[index]\n",
    "\n",
    "    def __len__(self):\n",
    "        return self.n_samples\n",
    "\n",
    "    \n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "dataset= ChatDataset()\n",
    "train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "58 58\n",
      "8 ['Buy', 'Car Brand', 'Car Type', 'Goodbye', 'Greeting', 'Noanswer', 'Purchasing', 'Thanks']\n"
     ]
    }
   ],
   "source": [
    "\n",
    "output_size=len(tags)\n",
    "input_size=len(X_train[0])\n",
    "print(input_size, len(all_words))\n",
    "print(output_size, tags)\n",
    "\n",
    "class NeuralNet(nn.Module):\n",
    "    def __init__(self, input_size, hidden_size, num_classes):\n",
    "        super(NeuralNet,self).__init__()\n",
    "        self.l1=nn.Linear(input_size, hidden_size)\n",
    "        self.l2=nn.Linear(hidden_size, hidden_size)\n",
    "        self.l3=nn.Linear(hidden_size, num_classes)\n",
    "\n",
    "        self.relu=nn.ReLU()\n",
    "\n",
    "    def forward(self,x ):\n",
    "        out =self.l1(x)\n",
    "        out=self.relu(out)\n",
    "        out=self.l2(out)\n",
    "        out=self.relu(out)\n",
    "        out =self.l3(out)\n",
    "        out=self.relu(out)\n",
    "\n",
    "        return out\n",
    "\n",
    "    \n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch100/1300, loss=1.4248\n",
      "epoch200/1300, loss=1.0136\n",
      "epoch300/1300, loss=1.4536\n",
      "epoch400/1300, loss=0.1091\n",
      "epoch500/1300, loss=0.0267\n",
      "epoch600/1300, loss=1.0504\n",
      "epoch700/1300, loss=0.5244\n",
      "epoch800/1300, loss=1.0475\n",
      "epoch900/1300, loss=0.5234\n",
      "epoch1000/1300, loss=0.5210\n",
      "epoch1100/1300, loss=1.0416\n",
      "epoch1200/1300, loss=0.5211\n",
      "epoch1300/1300, loss=0.5212\n",
      "final loss, loss=0.5212\n"
     ]
    }
   ],
   "source": [
    "\n",
    "model =NeuralNet(input_size, hidden_size, output_size).to(device)\n",
    "\n",
    "num_epochs=1300\n",
    "hidden_size=5\n",
    "lr=0.001\n",
    "batch_size=8\n",
    "\n",
    "criterion =nn.CrossEntropyLoss()\n",
    "optimizer= torch.optim.Adam(model.parameters(), lr=lr)\n",
    "for epoch in range (num_epochs):\n",
    "    for (words, labels) in train_loader:\n",
    "        \n",
    "        words= words.to(device)\n",
    "        labels= labels.to(dtype=torch.long).to(device)\n",
    "        optimizer.zero_grad()\n",
    "\n",
    "        outputs=model(words)\n",
    "        loss = criterion(outputs, labels)\n",
    "\n",
    "      \n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "    if (epoch +1) %100 ==0:\n",
    "     print(f'epoch{epoch+1}/{num_epochs}, loss={loss.item():.4f}')   \n",
    "    \n",
    "print(f'final loss, loss={loss.item():.4f}')   "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "training complete. file saved to data.pth\n"
     ]
    }
   ],
   "source": [
    "data = {\n",
    "\"model_state\": model.state_dict(),\n",
    "\"input_size\": input_size,\n",
    "\"hidden_size\": hidden_size,\n",
    "\"output_size\": output_size,\n",
    "\"all_words\": all_words,\n",
    "\"tags\": tags\n",
    "}\n",
    "\n",
    "FILE = \"data.pth\"\n",
    "torch.save(data, FILE)\n",
    "\n",
    "print(f'training complete. file saved to {FILE}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "interpreter": {
   "hash": "9d3a5badcae178cfb310f1a718d2f2a210d5ea6c7eb1bb151c756a2c589b19a8"
  },
  "kernelspec": {
   "display_name": "Python 3.9.7 64-bit ('base': conda)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
