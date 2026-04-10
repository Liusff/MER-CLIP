import torch
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import os
class TSNEVisualization:
    """t-SNE visualization for feature embeddings.
    
    Args:
        model (nn.Module): The model to extract features from.
        target_layer_name (str): Name of the layer from which features are extracted.
        n_components (int): Number of dimensions for t-SNE (usually 2 or 3).
        perplexity (float): Perplexity parameter for t-SNE.
        learning_rate (float): Learning rate for t-SNE.
        n_iter (int): Number of iterations for t-SNE.
    """

    def __init__(self, model: nn.Module, n_components=2, perplexity=30, learning_rate=200, n_iter=1000):
        self.model = model
        self.model.eval()
        self.target_activations = None
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter

        # Register hook
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register hook to capture activations from the specified layer."""
        def get_activations(module, input, output):
            self.target_activations = output.clone().detach()
        
        layer = 1
        prev_module = self.model
        #prev_module = prev_module.vision_encoder.transformer.resblocks[layer].ln_1
        prev_module = prev_module.vision_encoder.transformer.norm #.blocks[layer]

        target_layer = prev_module
        target_layer.register_forward_hook(get_activations)

    def extract_features(self, data: dict) -> torch.Tensor:
        """Pass data through the model to extract features."""
        with torch.no_grad():
            inputs = data['inputs']
            inputs = torch.stack(inputs)
            self.model(inputs=inputs.float().cuda(), mode='tensor')
        return self.target_activations

    def visualize(self, features: torch.Tensor, labels: torch.Tensor = None) -> None:
        """Apply t-SNE on the features and visualize."""
        #features_np = features.view(features.size(0), -1).cpu().numpy()
        
        # Perform t-SNE
        tsne = TSNE(n_components=self.n_components, perplexity=self.perplexity, learning_rate=self.learning_rate, n_iter=self.n_iter)
        features_tsne = tsne.fit_transform(features_np)

         # 对数据进行归一化操作
        x_min, x_max = np.min(features_tsne, 0), np.max(features_tsne, 0)
        features_tsne = features_tsne / (x_max - x_min)
        
        # Plot the t-SNE result
        plt.figure(figsize=(10, 8))
        if labels is not None:
            scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels.cpu().numpy(), cmap='viridis')
            plt.colorbar(scatter)
        else:
            plt.scatter(features_tsne[:, 0], features_tsne[:, 1])
        
        plt.title("t-SNE Visualization of Feature Embeddings")
        plt.xlabel("t-SNE Dim 1")
        plt.ylabel("t-SNE Dim 2")
        plt.savefig("tools/visualizations/vis/tSNE.jpg")
        plt.show()
    
    def visualize_tsne(self, features, labels, num_classes=7):
        """
        使用 t-SNE 将高维特征可视化为 2D 空间.
        
        参数:
            features: 从模型提取的特征，形状为 (N, D)，其中 N 是样本数，D 是特征维度.
            labels: 每个样本的标签，形状为 (N,) 或 (N, 1).
            num_classes: 类别数量，用于可视化中的颜色设置.
        """
        # 使用 t-SNE 将特征降维至 2D 空间
        tsne = TSNE(n_components=3, random_state=42)
        features_2d = tsne.fit_transform(features)
         # 对数据进行归一化操作
        x_min, x_max = np.min(features_2d, 0), np.max(features_2d, 0)
        features_2d = features_2d / (x_max - x_min)
        
        '''
        # 绘制 t-SNE 可视化
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        #sns.scatterplot(features_2d[:, 0], features_2d[:, 1], features_2d[:, 2], hue=labels, palette=palette, legend="full", alpha=0.7)
        
        ax.scatter(features_2d[:, 0], features_2d[:, 1], features_2d[:, 2], c=labels, cmap='viridis')
        ax.set_title("t-SNE Visualization of Features")
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.set_zlabel("t-SNE Dimension 3")
        plt.savefig("tools/visualizations/vis/tSNE_3d.jpg")
        plt.show()
        '''
        # 定义颜色映射
        unique_labels = np.unique(labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        color_map = {
            0: '#8A2BE2',  # 蓝紫色 (BlueViolet)
            1: 'deeppink',  # 靛青色 (Indigo)
            2: 'limegreen',  # 冷青色
            3: 'c',  # 板岩蓝色 (SlateBlue)
            4: '#FF4500',  # 橙红色 (特别区分)
            5: '#1E90FF',  # 道奇蓝色
            6: '#FFD700'   # 金色 (特别区分)
        }
        # 创建不同视角的图片
        angles = [(20, 45), (20, 90), (20, 135), (20, 180), (20, 225), (20, 270), (20, 315), (20, 360)]  # 各个视角的 (elev, azim) 值
        for j, (elev, azim) in enumerate(angles):  
            # 绘制 t-SNE 可视化
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            label_map = {"0": "anger", "1": "contempt", "2": "disgust", "3": "fear", "4": "happiness", "5": "sadness", "6": "surprise"}

            # 遍历每个类别，单独绘制并添加图例
            scatters = []
            for i, label in enumerate(unique_labels):
                scatter = ax.scatter(features_2d[labels == label, 0], 
                        features_2d[labels == label, 1], 
                        features_2d[labels == label, 2], 
                        color=color_map[label], label=f"{label_map[str(label)]}", alpha=0.7)
                scatters.append(scatter)
            # 设置视角
            ax.view_init(elev=elev, azim=azim)

            ax.set_title("t-SNE Visualization of Features")
            ax.set_xlabel("t-SNE Dimension 1")
            ax.set_ylabel("t-SNE Dimension 2")
            ax.set_zlabel("t-SNE Dimension 3")

            ax.legend()  # 显示图例
            save_path = "tools/visualizations/vis/tsne"
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            plt.savefig(os.path.join(save_path, f"tSNE_3d_angle_{j}.jpg"))
            plt.show()
            plt.close(fig)

    def __call__(self, data: dict, labels: torch.Tensor = None) -> None:
        """Main method to extract features and visualize with t-SNE."""
        features = self.extract_features(data)
        self.visualize(features, labels)
