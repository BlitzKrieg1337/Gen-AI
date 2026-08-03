print("Importing the embedding model")

import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

model = SentenceTransformer('all-MiniLM-L6-v2')

text = [
    # AI / Tech
    "Artificial intelligence models use vector embeddings to understand semantic meaning.",
    "OpneAI, Antropic and Google are in an AI race",
    "Machine learning systems can learn patterns from large datasets without explicit rules.",
    "Neural networks are inspired by the structure of the human brain.",

    # Food / Cooking
    "Whisk the egg whites until stiff peaks form before folding them into the batter.",
    "Add a pinch of salt to balance the sweetness in the dessert.",
    "Slow-cooked stews develop deeper flavor the longer they simmer.",
    "Fresh basil and garlic form the base of a classic pesto sauce.",

    # Space / Astronomy
    "The James Webb Space Telescope captured stunning images of a distant nebula.",
    "Golden sunlight filtered through the dense canopy of the redwood forest.",  # note: kept as your original "odd one out"
    "Astronomers discovered a new exoplanet orbiting a distant star.",
    "The rings of Saturn are made mostly of ice and rock particles.",

    # Finance
    "High-interest savings accounts help protect your money against inflation over time.",
    "Diversifying your investment portfolio reduces overall financial risk.",
    "Compound interest allows savings to grow faster over long time periods.",
    "Tracking your monthly expenses is the first step toward a healthy budget.",
]

def cosine_sim(a,b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot / (norm_a * norm_b)

embedding = model.encode(text)

kmeans = KMeans(n_clusters = 4, random_state = 42)
kmeans.fit(embedding)
labels = kmeans.labels_


print("1. Semantic search")
print("2. Cluster sentences")
print("3. Visualize clusters")
choice = input("Choose an option: ")

if choice == "1":
    prompt = input("Enter your prompt: ")
    embedding_prompt = model.encode(prompt)

    print("Finding a match...")
    result = []
    for i,e in enumerate(embedding):
        score = cosine_sim(embedding_prompt, e)
        result.append((score, text[i]))

    result.sort(reverse= True)

    print("Top 3 results")
    for score, sentence in result[:3]:
        print(sentence, " -> ", score)

elif choice == "2":
    pairs = list(zip(labels, text))
    pairs.sort()

    for label, sentence in pairs:
        print(f"Cluster {label  + 1} : {sentence}")

elif choice == "3":
    pca = PCA(n_components = 2)
    embedding_2d = pca.fit_transform(embedding)

    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c = labels)
    for i, txt in enumerate(text):
        plt.annotate(txt[:20], (embedding_2d[i, 0], embedding_2d[i, 1]))

    plt.show()