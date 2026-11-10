import pandas as pd
import re
from gensim.models import Word2Vec

# ---------- 1. Загрузка данных ----------
# Предполагается, что файл называется data.csv и содержит колонки text_id, full_text
df = pd.read_csv(
    'texts_olo_full.csv',
    encoding='utf-8',
    quotechar='"'      # чтобы корректно обрабатывать запятые внутри текста
)

# Берём только непустые тексты
texts = df['full_text'].dropna().astype(str).tolist()

# ---------- 2. Токенизация ----------
def tokenize(text: str) -> list[str]:
    """
    Простая токенизация:
    – приводит к нижнему регистру,
    – извлекает последовательности букв (включая š, ž, č и т.п.) и цифр.
    """
    text = text.lower()
    # \b\w+\b с флагом UNICODE корректно работает с диакритикой
    tokens = re.findall(r'\b\w+\b', text, flags=re.UNICODE)
    return tokens

# Каждый full_text превращаем в список токенов.
# Word2Vec будет рассматривать каждый такой список как одно «предложение».
sentences = [tokenize(t) for t in texts]

# ---------- 3. Обучение Word2Vec ----------
model = Word2Vec(
    sentences=sentences,
    vector_size=100,      # размерность вектора (для небольших корпусов можно 50–100)
    window=5,             # ширина контекстного окна
    min_count=5,          # слова с частотой < 5 игнорируются (уменьшите, если данных мало)
    workers=4,            # число потоков
    sg=1,                 # 1 = skip-gram (лучше для малых данных), 0 = CBOW
    epochs=20,            # количество проходов по корпусу
    seed=42               # для воспроизводимости
)

# ---------- 4. Сохранение ----------
# Полная модель (можно продолжить обучение)
model.save('word2vec_karelian.model')
# Только векторы слов (удобно для загрузки в других задачах)
model.wv.save('word2vec_karelian.kv')

print(f'Обучено слов: {len(model.wv)}')