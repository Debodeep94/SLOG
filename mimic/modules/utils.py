import numpy as np
import cv2
import torch
from surrogate import SurrogateModel, SurrogateLoss, CustomRidgeLoss, RidgeRegression
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from modules.utils import EarlyStopping


def penalty_builder(penalty_config):
    if penalty_config == '':
        return lambda x, y: y
    pen_type, alpha = penalty_config.split('_')
    alpha = float(alpha)
    if pen_type == 'wu':
        return lambda x, y: length_wu(x, y, alpha)
    if pen_type == 'avg':
        return lambda x, y: length_average(x, y, alpha)


def length_wu(length, logprobs, alpha=0.):
    """
    NMT length re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """

    modifier = (((5 + length) ** alpha) /
                ((5 + 1) ** alpha))
    return logprobs / modifier


def length_average(length, logprobs, alpha=0.):
    """
    Returns the average probability of tokens in a sequence.
    """
    return logprobs / length


def split_tensors(n, x):
    if torch.is_tensor(x):
        assert x.shape[0] % n == 0
        x = x.reshape(x.shape[0] // n, n, *x.shape[1:]).unbind(1)
    elif type(x) is list or type(x) is tuple:
        x = [split_tensors(n, _) for _ in x]
    elif x is None:
        x = [None] * n
    return x


def repeat_tensors(n, x):
    """
    For a tensor of size Bx..., we repeat it n times, and make it Bnx...
    For collections, do nested repeat
    """
    if torch.is_tensor(x):
        x = x.unsqueeze(1)  # Bx1x...
        x = x.expand(-1, n, *([-1] * len(x.shape[2:])))  # Bxnx...
        x = x.reshape(x.shape[0] * n, *x.shape[2:])  # Bnx...
    elif type(x) is list or type(x) is tuple:
        x = [repeat_tensors(n, _) for _ in x]
    return x


def generate_heatmap(image, weights):
    print(image[1:])
   # cv2.imshow('image',image[1:])
    print(image)
    print(f'image shape:{image.shape}')
    image = image.transpose(1,2,0) #1,2,0 was before I edited
    height, width, _ = image.shape
    weights = weights.reshape(int(weights.shape[0] ** 0.5), int(weights.shape[0] ** 0.5))
    weights = weights - np.min(weights)
    weights = weights / np.max(weights)
    weights = cv2.resize(weights, (width, height))
    weights = np.uint8(255 * weights)
    heatmap = cv2.applyColorMap(weights, cv2.COLORMAP_JET)
    result = heatmap * 0.5 + image * 0.5
    return result

def cycle(iterable):
    #https://github.com/pytorch/pytorch/issues/23900
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.delta = delta
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
def check_tensor_device(model):
    for name, tensor in model.named_parameters():
        print(f"Tensor: {name}, Device: {tensor.device}")

def surrogate_split(train_x, train_y, weights):
    train_x = train_x[:int(len(train_x)*0.8)]
    train_y = train_y[:int(len(train_y)*0.8)]

    val_x = train_x[int(len(train_x)*0.8):]
    val_y = train_y[int(len(train_y)*0.8):]

    weight_train= weights[:int(len(weights)*0.8)]

    return train_x, train_y, val_x, val_y, weight_train

def surrogate_regression(train_x, train_y, test_x, test_y, weights, id):
    test_list=[]
    for i in test_x: #initially it was test_x
        #i=i.cpu()
        new_list=i.tolist()
        test_list.append(new_list)
    test_list=np.array(test_list)

    train_list=[]
    for j in train_x:
        #i=i.cpu()
        new_list=j.tolist()
        train_list.append(new_list)
    train_list=np.array(train_list)

    print(len(train_list))
    #minmaxscaler
    scaler_train = MinMaxScaler()
    scaler_test = MinMaxScaler()
    # transform data
    train_x = scaler_train.fit_transform(train_list.reshape(len(train_list),512))

    test_x = scaler_test.fit_transform(test_list.reshape(len(test_list),512))
    
    train_x, train_y, val_x, val_y, weight_train = surrogate_split(train_x, train_y, weights)
    # Assume you have independent variables X and a dependent variable y
    train_x_pt = torch.tensor(train_x,dtype=torch.float) #pt: pytorch
    train_y_pt = torch.tensor(train_y, dtype=torch.float).reshape(-1,1) #pt:pytorch

    val_x_pt = torch.tensor(val_x,dtype=torch.float) #pt: pytorch
    val_y_pt = torch.tensor(val_y, dtype=torch.float).reshape(-1,1) #pt:pytorch

    test_x_pt = torch.tensor(test_x,dtype=torch.float)
    test_y_pt = torch.tensor(test_y, dtype=torch.float).reshape(-1,1)

    # Instantiate the model
    input_dim = train_x_pt.size(1)
    output_dim = train_y_pt.size(1)
    model_surr = SurrogateModel(input_dim, 3)
    # Define the loss function and optimizer
    criterion1 = torch.nn.MSELoss()
    criterion2 = SurrogateLoss()
    optimizer = torch.optim.Adam(model_surr.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    early_stopping = EarlyStopping(patience=50, delta=0)

    # Train the model
    num_epochs = 300
    train_loss = []
    validation_loss = []
    for epoch in range(num_epochs):
      y_hat = model_surr(train_x_pt)
      #print('yhat: ', y_hat)
      loss = criterion2(torch.tensor(weight_train), y_hat, train_y_pt)
      loss.backward()
      train_loss.append(loss.item())

      optimizer.step()
      optimizer.zero_grad()

      y_hat_val = model_surr(val_x_pt)
      val_loss = criterion1(y_hat_val, val_y_pt)
      validation_loss.append(val_loss.item())

      if ((epoch+1)%10)==0:
          print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item(), val_loss.item()))

      scheduler.step(val_loss)
      early_stopping(val_loss)

      if early_stopping.early_stop:
        print("Early stopping")
        break
    # Test the model
    with torch.no_grad():
        predicted = model_surr(test_x_pt)
        #print('Predicted values: ', predicted)

    #y_preds=np.asarray(scaler_test.inverse_transform(predicted))
    #gt=torch.tensor(scaler_test.inverse_transform(test_y_pt))
    #print(test_y)
    print('predicted', predicted)
    mse= criterion1(torch.tensor(predicted), test_y_pt)
    rel_rse_classical=torch.sum((torch.tensor(predicted)- test_y_pt)**2)/torch.sum((test_y_pt-torch.mean(test_y_pt))**2)
    rmse_classical=mse**0.5
    print('the rmse loss is ', rmse_classical)
    print('Saving surrogate model...')
    torch.save(model_surr, 'surrogate/split10k/'+'surr_lin_reg_new_split_'+str(id)+'.pt')
    print('variance of test list: ', torch.std(test_y_pt))
    print('variance of pred list: ', np.std(np.array(predicted)))
    print('relative RSE: ', rel_rse_classical)

    # Plot the curves
    fig_surr=plt.figure()
    plt.plot(train_loss, color= 'blue', label= 'train loss' )
    plt.plot(validation_loss, color= 'green', label= 'validation loss' )
    plt.legend()
    plt.title('Surrogate train on splitted training data')
    plt.savefig('plots/split10k/weight_lin_reg_new_split_B4_'+str(id)+'.png')
    plt.close(fig_surr)

    ############ Ridge regression ############

    # Define the hyperparameters
    input_size = train_x_pt.shape[1]
    alpha = 0.1
    epochs = 300
    lr = 0.02
    ridge_loss=[]
    ridge_val_loss = []
    # Initialize the Ridge Regression model
    model_ridge = RidgeRegression(input_size, alpha)

    # Define the loss function
    loss_fn = CustomRidgeLoss(alpha)

    # Define the optimizer
    optimizer = torch.optim.Adam(model_ridge.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    early_stopping = EarlyStopping(patience=100, delta=0)
    best_loss = float('inf')
    counter = 0  # Counter to track the number of epochs without improvement
    early_stopping = EarlyStopping(patience=100, delta=0)
    #scheduler = StepLR(optimizer, step_size=100, gamma=0.8)  # Reduce the learning rate by a factor of 0.5 every 20 epochs

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        outputs = model_ridge(train_x_pt)
        loss = loss_fn(model_ridge, torch.tensor(weight_train),outputs.squeeze(), train_y_pt)  # Remove extra dimensions
        ridge_loss.append(loss.item())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()
        with torch.no_grad():
            val_out=model_ridge(val_x_pt)
            val_loss=(criterion1(val_out.squeeze(), val_y_pt.reshape(-1)))
            ridge_val_loss.append(val_loss)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, val_loss: {val_loss.item():.4f}')
        scheduler.step(val_loss)
        early_stopping(val_loss)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    # Retrieve the learned coefficients
    coefficients = model_ridge.linear.weight.detach().numpy()
    intercept = model_ridge.linear.bias.detach().numpy()
    # Evaluate the model on the test set
    with torch.no_grad():
        test_outputs = model_ridge(test_x_pt)
    test_loss_ridge = (criterion1(test_outputs.squeeze(), test_y_pt.reshape(-1)))**0.5
    print('ridge outputs: ', test_outputs)
    print('test_loss: ', test_loss_ridge)
    rel_rse_ridge=torch.sum((torch.tensor(test_outputs)- test_y_pt)**2)/torch.sum((test_y_pt-torch.mean(test_y_pt))**2)
    print("rel_rse_ridge: ", rel_rse_ridge)
    torch.save(model_ridge, 'surrogate/split10k/'+'surr_ridge_reg_new_split_'+str(id)+'.pt')

    fig_ridge=plt.figure()
    plt.plot( ridge_loss, label='train')
    plt.plot( ridge_val_loss, label='test')
    plt.legend()
    plt.savefig('plots/split10k/weight_ridge_reg_new_split_B4_'+str(id)+'.png')
    plt.close(fig_ridge)
    return rmse_classical,rel_rse_classical, test_loss_ridge, rel_rse_ridge

def torch_minmax(input_tensor, device):
    loaded_data = torch.load('/home/debodeep.banerjee/R2Gen/train_min_max_scalar.pt')

    # Retrieve the min and max tensors
    min_tensor = loaded_data['min']
    max_tensor = loaded_data['max']

    min_tensor = min_tensor.to(device)
    max_tensor = max_tensor.to(device)

    # Use the retrieved tensors as needed
    #print(min_tensor.item(), max_tensor.item())

    # Apply min-max scaling to each sublist
    scaled_tensor = input_tensor.sub(min_tensor).div(max_tensor - min_tensor)


    # Apply the scaling function element-wise using torch.clamp to ensure values remain within the desired range
    scaled_tensor = torch.clamp(scaled_tensor, min=0, max=1)

    # Ensure the scaled tensor retains the gradient information for backpropagation
    scaled_tensor = scaled_tensor.clone().requires_grad_(True)
    return scaled_tensor
    
def convert(val):
    if val==0:
        return -1
    elif val == 1:
        return val
    else:
        return 0