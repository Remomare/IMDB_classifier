# IMDB_classifier

LSTM을 이용해 IMDB Sentimnet claasification을 수행했고
LSTM을 적용하는 법과
모듈화를 연습하는 것을 중심으로 코드를 작성해보았습니다

colab를 이용하여 GPU를 이용해
taining set으로는 정확도 97정도 나왔으며
test set으로는 88정도 까지나와 hyper_parameter를 조정하여 test set에서 acc를 더 높이도록 하려고 합니다.

모듈화를 시키면서 노트북으로만 실행하면 신경쓸 필요 없는 인자들이 넘거나는 과정을 레퍼런스 코드와 다르게 조정해야 했고 
그 과정을 겪으면서 코드을 더 잘 이해할 수 있었습니다

training 과정도 모듈화 하고 싶었으나 
epoch부분은 아직 어렵다고 느껴 메인 파일에 코드작성을 했습니다.
