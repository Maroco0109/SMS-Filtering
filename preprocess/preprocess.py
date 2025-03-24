from cProfile import label
import re
import random
import pandas as pd
import numpy as np
import warnings
from os.path import join as pjoin
warnings.filterwarnings(action='ignore')

'''
Description
-----------
전처리에 필요한 파라미터 지정
    args.use_test: 테스트 데이터 구축 여부, False일 때 test data는 빈 값 저장
    args.test_ratio: 테스트 데이터 비율
    args.seed: random seed 고정 (19로 고정)
'''
def base_setting(args):
    args.test_ratio = getattr(args, 'test_ratio', 0.2)
    args.seed = getattr(args, 'seed', 19)

'''
Description
-----------
전처리 함수

    def del_newline(text : str)
        개행/탭 문자 공백 문자로 변경
    def del_special_char(text : str)
        느낌표, 물음표, 쉼표, 온점, 물결, @을 제외한 특수문자 삭제
    def repeat_normalize(text : str, num_repeats : int)
        반복 문자 개수 num_repeats으로 제한
    def del_url(text : str)
        url을 'URL'로 대체
    def del_http(text : str)
        http 또는 https 삭제
    def del_duplicated_space(text : str)
        중복 공백 삭제
'''
def del_newline(text : str):
    return re.sub(r'[\s\n\t]+', ' ', text)

def del_special_char(text : str):
    # return re.sub(r'[^\w\s]', '', text)  # ✅ 모든 특수문자 제거 (괄호 포함)
    return re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)

repeatchars_pattern = re.compile(r'(\D)\1{2,}')
def repeat_normalize(text : str, num_repeats : int):
    if num_repeats > 0:
        text = repeatchars_pattern.sub(r'\1' * num_repeats, text)
    return text

def del_http(text : str):
    return re.sub(r'http(s){0,1}://', ' ', text)

url_regexp_1 = r'[a-zA-Z0-9_\-]+(\.[a-zA-Z0-9_-]+)+([a-zA-Z0-9.@?^=%&:/~+#-]*[a-zA-Z0-9_\-@?^=%&/~+#-]+)'
url_regexp_2 = r'http(s)*://[\w_\-]+(\.[a-zA-Z0-9_-]+)+([a-zA-Z0-9.@?^=%&:/~+#-]*[a-zA-Z0-9_\-@?^=%&/~+#-]+)'

def del_url(text):
    if "http" in text:
        return re.sub(url_regexp_2, " URL ", text)
    return re.sub(url_regexp_1, " URL ", text)

def del_duplicated_space(text : str):
    return re.sub(r'[\s]+', ' ', text)

# 민감 정보(이메일, 전화번호) 마스킹 함수
def mask_sensitive_info(text: str):
    # 이메일 주소를 <MASKED_EMAIL>로 치환
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '<MASKED_EMAIL>', text)
    # 전화번호(숫자 2~4자리-3~4자리-4자리 등)를 <MASKED_PHONE>로 치환
    text = re.sub(r'\b\d{2,4}[-.\s]?\d{3,4}[-.\s]?\d{4}\b', '<MASKED_PHONE>', text)
    return text

# 추가적인 불필요 패턴 정리 함수
def clean_extra_patterns(text: str):
    # 여러 밑줄을 하나의 공백으로 변경
    text = re.sub(r'_+', ' ', text)
    # 스팸 데이터셋의 @ifg 문자열 제거
    text = re.sub(r'ifg', '', text)
    # 불용 키워드 제거
    REMOVE_KEYWORDS = ["Web발신", "웹발신", "web발신", "국제발신","국외발신", "광고", "무료수신거부"]
    for keyword in REMOVE_KEYWORDS:
        text = re.sub(keyword, '', text)
    return text

def preprocess(text: str):
    # 민감 정보 먼저 마스킹 
    proc_txt = mask_sensitive_info(text)
    # URL 처리
    proc_txt = del_url(proc_txt)
    # 개행, 탭 등 공백 처리
    proc_txt = del_newline(proc_txt)
    # http 삭제
    proc_txt = del_http(proc_txt)
    # 특수문자 제거
    proc_txt = del_special_char(proc_txt)
    # 반복 문자 정규화
    proc_txt = repeat_normalize(proc_txt, num_repeats=3)
    # 중복 공백 제거
    proc_txt = del_duplicated_space(proc_txt)
    # 소문자 변환
    proc_txt = proc_txt.lower()
    # 추가 패턴 정리: 불필요한 밑줄, 특정 단어 제거 등
    proc_txt = clean_extra_patterns(proc_txt)
    # 양쪽 공백 제거
    proc_txt = proc_txt.strip()
    return proc_txt

def processing(args, data, is_test=False):
    base_setting(args)

    # seed 고정
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    if not is_test:
        # labeling
        labels = ['spam', 'ham']
        data['label'] = list(map(lambda x: labels.index(x), data['label']))
    
    # text processing
    data['proc_text'] = list(map(preprocess, data['text']))
    return data


'''
Description
-----------
테스트 시, 되도록 모든 클래스에 대한 정답률 파악을 위해 \
    5개 미만의 데이터를 보유한 클래스의 경우 임의로 할당

train, valid, test의 클래스 분포를 기존 data의 클래스 분포와 동일하게 유지  
'''
def get_balanced_dataset(data : pd.DataFrame, test_ratio : float, use_test : bool):
    
    valid, train, test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for idx in data.label.unique().tolist():
        sub_data = data[data.label==idx]
        num_valid = num_test = int(len(sub_data) * test_ratio)

        if num_test == 0:
            if len(sub_data) < 2:
                train = pd.concat([train, sub_data], ignore_index=True)
                continue
            elif len(sub_data) < 3:
                test = pd.concat([test, sub_data.iloc[:1]], ignore_index=True)
                train = pd.concat([train, sub_data.iloc[1:]], ignore_index=True)
                continue
            else: 
                # class 내 데이터가 3-4개 인 경우
                num_valid = num_test = int(len(sub_data)/ 3)

        test_idx = num_test if use_test else 0
        valid_idx = 2 * num_valid if use_test else num_valid

        test = pd.concat([test, sub_data.iloc[:test_idx]], ignore_index=True)
        valid = pd.concat([valid, sub_data.iloc[test_idx:valid_idx]], ignore_index=True)
        train = pd.concat([train, sub_data.iloc[valid_idx:]], ignore_index=True)

        del sub_data

    return train, test, valid

'''
Description
-----------
전체 데이터를 train, valid, test로 분할하여 args.save_dir 내에 저장
'''
def split_dataset(args, data):
    train, test, valid = get_balanced_dataset(data=data, test_ratio=args.test_ratio, \
        use_test=args.use_test)

    print(f"Train Distribution : \n{train.label.value_counts()}")
    print(f"Valid Distribution : \n{valid.label.value_counts()}")
    print(f"Test Distribution : \n{test.label.value_counts()}")
    
    train = train.sample(frac=1).reset_index(drop=True)
    valid = valid.sample(frac=1).reset_index(drop=True)
    test = test.sample(frac=1).reset_index(drop=True)

    valid.to_csv(pjoin(args.save_dir, 'valid.csv'), index=False)
    test.to_csv(pjoin(args.save_dir, 'test.csv'), index=False)
    train.to_csv(pjoin(args.save_dir, 'train.csv'), index=False)

    print(f"Total Number of Data : {len(data)} -> {len(valid) + len(test) + len(train)}")
    return
