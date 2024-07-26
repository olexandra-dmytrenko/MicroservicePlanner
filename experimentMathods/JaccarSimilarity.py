from sklearn.metrics import jaccard_score
import numpy as np

#приводить до бінарного вигляду, а тоді шукає бінарну різницю зважуючи на схожість
# Мікросервіси з числовими характеристиками
microservices_numeric = np.array([
    [3, 5, 2, 0, 7, 0],  # Мікросервіс A
    [1, 0, 3, 4, 0, 1],  # Мікросервіс B
    [0, 2, 0, 5, 2, 0]   # Мікросервіс C
])

# Встановлюємо поріг для бінаризації
threshold = 2

# Перетворюємо числові дані на бінарні
microservices_binary = (microservices_numeric >= threshold).astype(int)
print(str(microservices_binary))
# Обчислення коефіцієнта Жаккара між мікросервісом A і B
jaccard_index_AB = jaccard_score(microservices_binary[0], microservices_binary[1])

# Обчислення коефіцієнта Жаккара між мікросервісом A і C
jaccard_index_AC = jaccard_score(microservices_binary[0], microservices_binary[2])

# Обчислення коефіцієнта Жаккара між мікросервісом B і C
jaccard_index_BC = jaccard_score(microservices_binary[1], microservices_binary[2])

print(f"Jaccard Similarity between A and B: {jaccard_index_AB}")
print(f"Jaccard Similarity between A and C: {jaccard_index_AC}")
print(f"Jaccard Similarity between B and C: {jaccard_index_BC}")
