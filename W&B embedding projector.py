import numpy as np
import wandb
from sklearn.decomposition import PCA

# 使用 API Key 登录 W&B
wandb.login(key="37b425ed9e2e3bcd4fb88bf7689251942fec6029")  # 请用你的实际 API Key 替换这里

# 1. 加载 .npy 文件中的嵌入和标签
embeddings = np.load("./data/cifar10_embeddings.npy")  # 替换为你的 embeddings 文件路径
labels = np.load("./data/cifar10_labels.npy")          # 替换为你的 labels 文件路径

print(f"Embedding shape: {embeddings.shape}")
print(f"Labels shape: {labels.shape}")

# 检查数据的一致性
assert len(embeddings) == len(labels), "Embeddings 和 Labels 的长度不匹配"

# 2. 初始化 W&B，确保项目名与在 W&B 上创建的项目名称一致
wandb.init(project="image_embedding", entity="nlpkerryspace")  # 确保项目名和 entity 正确

# 3. 使用 PCA 将嵌入降维到 3D 空间
use_pca = True  # 如果不需要降维，可以将其设置为 False
if use_pca:
    print("Applying PCA to reduce dimensionality to 3D...")
    pca = PCA(n_components=3)
    embeddings_reduced = pca.fit_transform(embeddings)
else:
    embeddings_reduced = embeddings  # 如果不降维，直接使用原始嵌入

# 上传到 W&B
# 创建一个包含嵌入和标签的 numpy 数组，这里用 dtype=object 确保正确的数组格式
embedding_with_labels = np.hstack((embeddings_reduced, labels.reshape(-1, 1)))

# 上传嵌入数据到 W&B 3D 可视化
wandb.log({"3D_embeddings": wandb.Object3D(embedding_with_labels)})

# 结束 W&B 运行
wandb.finish()

print("3D embedding upload completed! Check your W&B project page for 3D visualization.")
