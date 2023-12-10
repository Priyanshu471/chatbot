
#  Sample Dataset for Training a Chatbot
college_dataset = [
    {"intent": "admission", "patterns": ["How to apply for admission?", "Admission process"]},
    {"intent": "courses", "patterns": ["Tell me about available courses", "Course details"]},
    {"intent": "fees", "patterns": ["What are the tuition fees?", "Fee structure"]},
]

# Extract words and intents from the dataset
all_words = []
tags = []
xy = []

for entry in college_dataset:
    intent = entry["intent"]
    tags.append(intent)
    for pattern in entry["patterns"]:
        words = tokenize(pattern)
        all_words.extend(words)
        xy.append((words, intent))

# Stem and preprocess words
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Prepare training data
X_train = []
y_train = []

for (pattern_words, intent) in xy:
    bag = bag_of_words(pattern_words, all_words)
    X_train.append(bag)
    label = tags.index(intent)
    y_train.append(label)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
