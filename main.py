import argparse
from modelBase import Seq2SeqModel
from AttentionSeq2SeqModel import AttentionSeq2SeqModel
parser = argparse.ArgumentParser()

parser.add_argument('--task', default='train',help='train |validate| test | others')
parser.add_argument('--model_name',type=str,default='seq2seq',help='seq2seq|biLSTMseq2seq|hybridseq2seq')
parser.add_argument('--dataset_dir', default='./wikihow summarization/dataset',help='url dataset')
parser.add_argument('--model_save_dir',default='./model/general')
parser.add_argument('--best_model_save_dir',default='./model/general')
parser.add_argument('--example',default='./text.txt')

#Siêu tham số
parser.add_argument('--max_text_len', type=int,default=1000)
parser.add_argument('--max_summary_len', type=int,default=100)
parser.add_argument('--embedding_dim', type=int,default=200)
parser.add_argument('--latent_dim', type=int,default=300)
parser.add_argument('--dropout_rate', type=float,default=0.4)
parser.add_argument('--recurrent_dropout_rate', type=float,default=0.0)
parser.add_argument('--batch_size', type=int,default=32)
parser.add_argument('--optimizer',default='adam')
parser.add_argument('--loss',default='sparse_categorical_crossentropy')
parser.add_argument('--epoch', type=int,default=50)
parser.add_argument('--attention_model', type=str,default='dot')
parser.add_argument('--encode_layer', type=int,default=3)
parser.add_argument('--decode_layer', type=int,default=1)
parser.add_argument('--rnn_cell', type=str,default='lstm',help='lstm|gru')
parser.add_argument('--earlystop', type=bool,default=False)

args = parser.parse_args()
def main():
    if args.model_name=='seq2seq':
        model=Seq2SeqModel(args=args)
    elif args.model_name=='attentionseq2seq':
        model=AttentionSeq2SeqModel(args=args)
    else:
        print("Model isn't existed")
        return 0    
    model.preprocess()
    model.bulid_model()
    if args.task=='example':
        with open(args.example,'r') as f:
            example=f.read()
        summary=model.summary(example)
        with open(args.example,'a') as f:
            f.write("Summary:",summary)
    if args.task=='train':
        model.train()
    if args.task=='test':
        model.test()

if __name__ == "__main__":
    main()