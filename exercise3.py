import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist, pos_tag
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

with open('moby_dick.txt', 'r') as file:
    moby_dick_text = file.read()

tokens = word_tokenize(moby_dick_text)

stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

pos_tags = pos_tag(filtered_tokens)

pos_counts = FreqDist(tag for (word, tag) in pos_tags)
top_pos = pos_counts.most_common(5)
print('Most common parts of speech:')
for pos, count in top_pos:
    print(pos, ':', count)


lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word, pos=pos[0].lower()) for word, pos in pos_tags][:20]

pos_labels, pos_values = zip(*pos_counts.items())
plt.figure(figsize=(10, 5))
plt.bar(pos_labels, pos_values)
plt.xlabel('Parts of Speech')
plt.ylabel('Frequency')
plt.title('Frequency Distribution of Parts of Speech')
plt.show()