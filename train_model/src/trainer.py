import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ==================== КОНФИГУРАЦИЯ ====================
CONFIG = {
    'random_state': 42,
    'test_size': 0.1,
    'model_dir': Path("saved_model"),
    'patience': 10,
    'epochs': 200,
    'learning_rate': 0.001,
    'thresholds': [0.5, 0.7, 0.75, 0.8, 0.85],
    'batch_norm': True,
    'dropout_rates': [0.4, 0.3]
}


# ==================== АРХИТЕКТУРА МОДЕЛИ ====================
class CreditScoringModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        layers = [
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256) if CONFIG['batch_norm'] else nn.Identity(),
            nn.Dropout(CONFIG['dropout_rates'][0]),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128) if CONFIG['batch_norm'] else nn.Identity(),
            nn.Dropout(CONFIG['dropout_rates'][1]),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
            nn.Sigmoid()
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# ==================== ФУНКЦИИ ДЛЯ ОБУЧЕНИЯ ====================
def prepare_data(df):
    """Подготовка и разделение данных"""
    y = df["loan_status"]
    X = df.drop(['loan_status'], axis=1)
    return train_test_split(
        X, y,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state']
    )


def train_model(model, X_train, y_train, X_val, y_val):
    """Процесс обучения модели"""
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.BCELoss()

    best_metrics = {th: {'f1': 0, 'model_state': None} for th in CONFIG['thresholds']}
    best_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(CONFIG['epochs']):
        # Training phase
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_probs = val_outputs.cpu().numpy().flatten()

            # Update best models for each threshold
            for threshold in CONFIG['thresholds']:
                val_preds = (val_probs > threshold).astype(int)
                current_f1 = f1_score(y_val.cpu().numpy(), val_preds)

                if current_f1 > best_metrics[threshold]['f1']:
                    best_metrics[threshold]['f1'] = current_f1
                    best_metrics[threshold]['model_state'] = model.state_dict()

        # Early stopping logic
        if val_loss < best_loss - 0.001:
            best_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= CONFIG['patience']:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    return best_metrics


def save_artifacts(model, scaler, feature_names, best_metrics):
    """Сохранение всех артефактов модели"""
    CONFIG['model_dir'].mkdir(exist_ok=True)

    # Save scaler and feature names
    joblib.dump(scaler, CONFIG['model_dir'] / 'scaler.pkl')
    with open(CONFIG['model_dir'] / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f)

    # Save best models for each threshold
    for threshold, metrics in best_metrics.items():
        if metrics['model_state'] is not None:
            torch.save({
                'model_state_dict': metrics['model_state'],
                'input_size': model.layers[0].in_features,
                'threshold': threshold,
                'f1_score': metrics['f1']
            }, CONFIG['model_dir'] / f'model_th_{threshold}.pth')

    # Save TorchScript model
    model_scripted = torch.jit.script(model)
    model_scripted.save(CONFIG['model_dir'] / 'model_scripted.pt')

    # Save config
    config = {
        'input_size': model.layers[0].in_features,
        'feature_names': feature_names,
        'best_thresholds': {str(th): m['f1'] for th, m in best_metrics.items()},
        'training_config': CONFIG
    }
    with open(CONFIG['model_dir'] / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Оценка модели на тестовых данных"""
    model.eval()
    with torch.no_grad():
        probs = model(X_test).cpu().numpy().flatten()
        preds = (probs > threshold).astype(int)
        y_np = y_test.cpu().numpy().flatten()

        print(f"\nEvaluation Report (threshold={threshold}):")
        print(classification_report(y_np, preds))

        # Confusion matrix visualization
        cm = confusion_matrix(y_np, preds)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix (threshold={threshold})')
        plt.show()


# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================
def main():
    # 1. Загрузка и подготовка данных
    df = pd.read_csv('your_data.csv')  # Замените на ваш путь к данным
    X_train, X_test, y_train, y_test = prepare_data(df)
    feature_names = X_train.columns.tolist()

    # 2. Масштабирование данных
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Преобразование в тензоры
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1).to(device)

    # 4. Инициализация и обучение модели
    model = CreditScoringModel(input_size=X_train.shape[1]).to(device)
    print(f"Training on {device}...")
    best_metrics = train_model(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)

    # 5. Сохранение артефактов
    save_artifacts(model, scaler, feature_names, best_metrics)
    print(f"\nAll artifacts saved to: {CONFIG['model_dir'].absolute()}")

    # 6. Оценка модели
    evaluate_model(model, X_test_tensor, y_test_tensor)


if __name__ == "__main__":
    main()