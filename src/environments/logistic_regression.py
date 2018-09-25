import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class LogisticRegression(object):
    """
        Logistic regression benchmark which resembles the benchmark used in the MTBO / Freeze Thaw paper.
        The hyperparameters are the learning rate (on log scale), L2 regularization, batch size and the
        dropout ratio on the inputs.
        The weights are optimized with stochastic gradient descent and we do NOT perform early stopping on
        the validation data set.
    """

    def __init__(self, num_epochs=100):
        self.num_epochs = num_epochs

    def __call__(self, x):
        return self.objective_function(x)

    def objective_function(self, x):
        """rounding and call fit
        """

        learning_rate = float(10 ** x[0])
        l2_reg = float(x[1])
        batch_size = int(x[2])
        dropout_rate = float(x[3])

        lc_curve = self.run(learning_rate=learning_rate,
                            l2_reg=l2_reg,
                            batch_size=batch_size,
                            dropout_rate=dropout_rate,
                            num_epochs=self.num_epochs)
        y = lc_curve[-1]
        return y

    @staticmethod
    def get_meta_information():
        return {'name': 'Logistic Regression',
                'bounds': [[-6, 0],    # learning rate
                           [0, 1],     # l2 regularization
                           [20, 2000], # batch size
                           [0, .75]],  # dropout rate
                'f_opt': None,
                }

    def run(self, learning_rate=0.1, l2_reg=0.0,
            batch_size=32, dropout_rate=0, num_epochs=100):
        """training, predict, return validation error (and time?)
        """

        # Hyper Parameters 
        input_size = 784
        num_classes = 10

        # MNIST Dataset (Images and Labels)
        train_dataset = dsets.MNIST(root='.', 
                                    train=True, 
                                    transform=transforms.ToTensor(),
                                    download=True)

        test_dataset = dsets.MNIST(root='.', 
                                train=False, 
                                transform=transforms.ToTensor())

        # Dataset Loader (Input Pipline)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                batch_size=batch_size, 
                                                shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                batch_size=batch_size, 
                                                shuffle=False)
        
        # Model
        model = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(input_size, num_classes),
        ).to(device)

        # Loss and Optimizer
        # Softmax is internally computed.
        # Set parameters to be updated.
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2_reg) 

        # Training the Model
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = Variable(images.view(-1, 28*28)).to(device)
                labels = Variable(labels).to(device)

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if (i+1) % 100 == 0:
                    print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f' 
                        % (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data.item()))

            # Test the Model
            correct = 0
            total = 0
            learning_curve = []
            for images, labels in test_loader:
                images = Variable(images.view(-1, 28*28)).to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            learning_curve.append(correct.cpu().numpy() / total)
            print("validation", learning_curve[-1])

        return learning_curve
