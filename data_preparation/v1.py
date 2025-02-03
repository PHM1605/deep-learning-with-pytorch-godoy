x_train_tensor = torch.as_tensor(x_train).float()
y_train_tensor = torch.as_tensor(y_train).float()
train_data = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset = train_data, batch_size=16, shuffle=True)