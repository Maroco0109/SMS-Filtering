# config.py

# Hyperparameters
class Config:
    MAX_LEN = 128
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 3

# Label
class Label:
    labels = {
    "주식/투자": ["주식", "투자", "수익", "펀드"],
    "도박": ["카지노", "베팅", "스포츠토토"],
    "대출": ["대출", "저금리", "금리", "즉시"],
    "통신가입": ["통신사", "가입", "할인", "이벤트"],
    "대리운전": ["대리운전", "운전", "기사"],
    "광고": ["광고", "홍보", "프로모션"],
}