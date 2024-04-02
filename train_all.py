from src.classify.utils.devices_selector import get_recommended_device
from src.classify.equal_symbol.train_equal_symbol import train_equal_symbol
from src.classify.operator.train_operator import train_operator
from src.classify.digit.train_digit_all import train_digit_all

if __name__ == '__main__':
    device = get_recommended_device()

    train_equal_symbol(device)
    train_operator(device)
    train_digit_all(device)
