from typing import Tuple
from neural import *
from sklearn.model_selection import train_test_split

def parse_line(line: str) -> Tuple[List[float], List[float]]:
    """Splits line of CSV into inputs and output (transormfing output as appropriate)

    Args:
        line - one line of the CSV as a string

    Returns:
        tuple of input list and output list
    """
    tokens = line.split(",")
    out = int(tokens[0])
    output = [0 if out == 1 else 0.5 if out == 2 else 1]

    inpt = [float(x) for x in tokens[1:]]
    return (inpt, output)


def normalize(data: List[Tuple[List[float], List[float]]]):
    """Makes the data range for each input feature from 0 to 1

    Args:
        data - list of (input, output) tuples

    Returns:
        normalized data where input features are mapped to 0-1 range (output already
        mapped in parse_line)
    """
    new_dict = {}
    for key, value in data_dict.items():
        if isinstance(key, str):
            if key.lower() in ['male', 'm']:
                new_dict[1] = value
            elif key.lower() in ['female', 'f']:
                new_dict[0] = value
            else:
                new_dict[key] = value
        else:
            new_dict[key] = value
        if isinstance(key, str):
            if key.lower() in ['group a']:
                new_dict[0] = value
        elif isinstance(key, str):
            if key.lower() in ['group b']:
                new_dict[1] = value
        elif isinstance(key, str):
            if key.lower() in ['group c']:
                new_dict[2] = value
        elif isinstance(key, str):
            if key.lower() in ['group d']:
                new_dict[3] = value
        elif isinstance(key, str):
            if key.lower() in ['group e']:
                new_dict[4] = value
    return new_dict


with open("study_performance.csv", "r") as f:
    training_data = [parse_line(line) for line in f.readlines() if len(line) > 4]

# print(training_data)
td = normalize(training_data)
# print(td)

train, test = train_test_split(td)

nn = NeuralNet(13, 3, 1)
nn.train(train, iters=10000, print_interval=1000, learning_rate=0.2)

for i in nn.test_with_expected(test):
    difference = round(abs(i[1][0] - i[2][0]), 3)
    print(f"desired: {i[1]}, actual: {i[2]} diff: {difference}")
