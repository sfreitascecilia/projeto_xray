import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Label, Button
import os


# Definição do modelo CNN
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Ajuste do tamanho da camada Linear após Flatten
        self._to_linear = None
        self._calculate_flattened_size((3, 150, 150))  # (C, H, W)

        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def _calculate_flattened_size(self, shape):
        # Realiza um forward pass pela parte de features para calcular o tamanho após Flatten
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            dummy_output = self.features(dummy_input)
            self._to_linear = dummy_output.numel()  # Número total de elementos após Flatten

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Dataset personalizado para carregar imagens de um único diretório
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, label_map=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        self.label_map = label_map or {}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        label = self.label_map.get(self.image_files[idx], 0)
        if self.transform:
            image = self.transform(image)
        return image, label


# Função para treinar e salvar o modelo
def train_and_save_model(train_dir, validation_dir, model_path, img_height=150, img_width=150, batch_size=32,
                         epochs=10):
    # Definição das transformações de imagem
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Mapeamento das imagens para os rótulos
    label_map = {}
    for i, filename in enumerate(os.listdir(train_dir)):
        label_map[filename] = 1 if 'pneumonia' in filename.lower() else 0

    # Preparação dos datasets
    train_dataset = CustomImageDataset(image_dir=train_dir, transform=transform, label_map=label_map)
    validation_dataset = CustomImageDataset(image_dir=validation_dir, transform=transform, label_map=label_map)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    # Inicialização do modelo, critério e otimizador
    model = CNNModel()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Treinamento do modelo
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Época {epoch + 1}/{epochs}, Perda: {epoch_loss:.4f}')

    # Salvar o modelo treinado
    torch.save(model.state_dict(), model_path)


# Função para carregar e pré-processar uma imagem
def preprocess_image(img_path, img_height=150, img_width=150):
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)
    return img


# Função para fazer a previsão com a imagem
def predict_image(img_path, model):
    model.eval()
    img_array = preprocess_image(img_path)
    with torch.no_grad():
        outputs = model(img_array)
        prediction = outputs.item()
    return "Pneumonia" if prediction > 0.5 else "Normal"


# Função para exibir a imagem e o resultado usando tkinter
def show_image_and_result(image_path, result):
    root = tk.Tk()
    root.title("Resultado da Análise")

    # Exibir a imagem
    img = Image.open(image_path)
    img = img.resize((400, 400), Image.LANCZOS)  # Usar Image.LANCZOS no lugar de Image.ANTIALIAS
    img_tk = ImageTk.PhotoImage(img)
    img_label = Label(root, image=img_tk)
    img_label.pack()

    # Exibir o resultado
    result_label = Label(root, text=f"Resultado da análise: {result}", font=("Helvetica", 16))
    result_label.pack()

    # Adicionar um botão para fechar a aplicação
    close_button = Button(root, text="Fechar", command=root.destroy)
    close_button.pack()

    root.mainloop()


# Caminhos para os diretórios de treinamento e validação
train_dir = 'treinamento'
validation_dir = 'validacao'
model_path = 'modelo_raio_x.pth'

# Treinar e salvar o modelo
train_and_save_model(train_dir, validation_dir, model_path)

# Inicializar o modelo e carregar os pesos salvos
model = CNNModel()
model.load_state_dict(torch.load(model_path, weights_only=True))  # Especificar weights_only=True

# Caminho da imagem para análise
image_path = 'imagem/imagem_raio_x_1.jpg'  # Normal
# image_path = 'imagem/imagem_raio_x_2.jpg'  # Com pneumonia

# Fazer a previsão
result = predict_image(image_path, model)
print(f"A imagem foi classificada como: {result}")

# Exibir a imagem e o resultado
show_image_and_result(image_path, result)
