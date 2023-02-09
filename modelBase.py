from tensorflow.keras.layers import Input, LSTM,GRU, Embedding, Dense, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from datasets import load_from_disk 
import numpy as np
import os
import json
from evalution import ROUGE

class Seq2SeqModel(object):
    def __init__(
        self,
        args,       
    ):
        self.args=args

    def preprocess(self):
        dataset=load_from_disk(self.args.dataset_dir)
        self.x_train=pad_sequences(dataset['train']['text_seq'],maxlen=self.args.max_text_len,padding='post')
        self.x_val=pad_sequences(dataset['valid']['text_seq'],maxlen=self.args.max_text_len,padding='post')
        self.x_test=pad_sequences(dataset['test']['text_seq'],maxlen=self.args.max_text_len,padding='post')
        self.y_train=pad_sequences(dataset['train']['summary_seq'],maxlen=self.args.max_summary_len,padding='post')
        self.y_val=pad_sequences(dataset['valid']['summary_seq'],maxlen=self.args.max_summary_len,padding='post')
        self.y_test=pad_sequences(dataset['test']['summary_seq'],maxlen=self.args.max_summary_len,padding='post')
        self.index2word_sum=dataset['index2word_summary']['word']
        self.index2word_text=dataset['index2word_text']['word']
        self.word2index_sum={w:i for i,w in enumerate( self.index2word_sum)}
        self.word2index_text={w:i for i,w in enumerate( self.index2word_text)} 
        self.val_summary=dataset['valid']['summary']
        self.test_summary=dataset['test']['summary']
        self.x_vocab_size=len(self.word2index_text)
        self.y_vocab_size=len(self.word2index_sum)

    def bulid_model(self):
                # model
        encoder_inputs = Input(shape=(self.args.max_text_len,))
        encoder_output =  Embedding(self.x_vocab_size, self.args.embedding_dim,trainable=True,mask_zero=True)(encoder_inputs)
        
        for i in range(self.args.encode_layer):
            if self.args.rnn_cell=='lstm':
                encoder_layer = LSTM(self.args.latent_dim,return_sequences=True,return_state=True,dropout=self.args.dropout_rate,recurrent_dropout=self.args.recurrent_dropout_rate)
            elif self.args.rnn_cell=='gru':
                encoder_layer = GRU(self.args.latent_dim,return_sequences=True,return_state=True,dropout=self.args.dropout_rate,recurrent_dropout=self.args.recurrent_dropout_rate)
            else:
                raise Exception("rnn cell is only lstm or gru")
            encoder_output, *encoder_state = encoder_layer(encoder_output)
        

        decoder_inputs = Input(shape=(None,))
        dec_emb_layer = Embedding(self.y_vocab_size, self.args.embedding_dim,trainable=True,mask_zero=True)
        dec_emb = dec_emb_layer(decoder_inputs)
        decoder_outputs=dec_emb
        list_decoder=[]
        for i in range(self.args.decode_layer):
            if self.args.rnn_cell=='lstm':
                decoder_layer = LSTM(self.args.latent_dim,return_sequences=True,return_state=True,dropout=self.args.dropout_rate,recurrent_dropout=self.args.recurrent_dropout_rate)
            elif self.args.rnn_cell=='gru':
                decoder_layer = GRU(self.args.latent_dim,return_sequences=True,return_state=True,dropout=self.args.dropout_rate,recurrent_dropout=self.args.recurrent_dropout_rate)
            list_decoder.append(decoder_layer)
            decoder_outputs, *decode_state = decoder_layer(decoder_outputs,initial_state=encoder_state)
        decoder_dense =  TimeDistributed(Dense(self.y_vocab_size, activation='softmax'))
        decoder_outputs = decoder_dense(decoder_outputs) 
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.model.compile(optimizer=self.args.optimizer, loss=self.args.loss)

        #Encoder model
        self.encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_output]+encoder_state)
        # Decoder model
        decoder_state_input = [Input(shape=(self.args.latent_dim,))]
        if self.args.rnn_cell=='lstm':
            c = Input(shape=(self.args.latent_dim,))
            decoder_state_input.append(c)
        # decoder_hidden_state_input = Input(shape=(args.max_text_len,args.latent_dim))
        decoder_outputs2= dec_emb_layer(decoder_inputs) 

        for layer in list_decoder:
            decoder_outputs2, *state = layer(decoder_outputs2, initial_state=decoder_state_input)
        decoder_outputs2 = decoder_dense(decoder_outputs2) 
        self.decoder_model = Model(
            [decoder_inputs] + decoder_state_input,
            [decoder_outputs2]+state)

    def train(self):
        current_epoch=0
        if os.path.exists(os.path.join(self.args.model_save_dir,'history.json'))==False:
            open(os.path.join(self.args.model_save_dir,'history.json'), "x")
        with open(os.path.join(self.args.model_save_dir,'history.json'),'r') as f:
            val_loss=[i['val_loss'][0] for i in f.read().splitlines()]
            current_epoch=len(val_loss)
            val_loss=[1<<16,1<<16,1<<16,1<<16]+val_loss
        best_loss=1<<16
        if current_epoch!=0:
            self.model.load_weights(os.path.join(self.args.model_save_dir,'current_model_weight'))
            with open(os.path.join(self.args.best_model_save_dir,'loss_best_model.json'),'r') as f:
                x=json.loads(f.read())
                best_loss=x['val_loss'][0]
        for i in range(current_epoch,self.args.epoch):
            if self.args.earlystop and val_loss[-1]>val_loss[-2] and val_loss[-1]>val_loss[-2] and val_loss[-1]>val_loss[-2]:
                print(f"Model early stop at epoch {i}")
                return 0
            print(f'Train epoch {i+1}/{self.args.epoch}:')
            his=self.model.fit([self.x_train,self.y_train[:,:-1]], self.y_train.reshape(self.y_train.shape[0],self.y_train.shape[1], 1)[:,1:] ,epochs=1,batch_size=self.args.batch_size, validation_data=([self.x_val,self.y_val[:,:-1]], self.y_val.reshape(self.y_val.shape[0],self.y_val.shape[1], 1)[:,1:]))
            self.model.save_weights(os.path.join(self.args.model_save_dir,'current_model_weight'))
            with open(os.path.join(self.args.model_save_dir,'history.json'),'a') as f:
                f.write(json.dumps(his.history))
                f.write('\n')
            val_loss.append(his.history['val_loss'][0])
            if best_loss>his.history['val_loss'][0]:
                best_loss=his.history['val_loss'][0]
                with open(os.path.join(self.args.best_model_save_dir,'loss_best_model.json'),'w') as f:
                    f.write(json.dumps(his.history))
                self.model.save_weights(os.path.join(self.args.best_model_save_dir,'best_model_weight'))
            
        return 0

    def predict(self,input_seq):
        e_out, *e = self.encoder_model.predict(input_seq)

        target_seq = np.ones((input_seq.shape[0],1))*self.word2index_sum['<SOS>']
        pad_id=self.word2index_sum['<PAD>']
        eos_id=self.word2index_sum['<EOS>']
        res=np.ones((input_seq.shape[0],1))*self.word2index_sum['<SOS>']
        # start_arr=np.ones((input_seq.shape[0],1))*self.word2index_sum['<SOS>']
        # option 1
        # for i in range(self.args.max_summary_len-2):
        #     output_tokens, *hc = self.decoder_model.predict([target_seq]+e)
        #     sampled_token_index = np.argmax(output_tokens,axis =-1)
        #     # sampled_token = self.index2word_sum[sampled_token_index]
        #     target_seq = np.expand_dims(np.squeeze(sampled_token_index), axis=1)
        #     res=np.concatenate((res,target_seq),axis=1)
        #     e = hc
        # not_eos_seq=target_seq-np.ones(target_seq.shape)*eos_id
        # not_eos_seq=np.array(not_eos_seq,dtype=bool)
        # not_pad_seq=target_seq-np.ones(target_seq.shape)*pad_id
        # not_pad_seq=np.array(not_pad_seq,dtype=bool)        
        # eos_mask=not_pad_seq*not_eos_seq
        # target_seq=np.ones(eos_mask.shape)*eos_id*eos_mask+pad_id*np.ones(eos_mask.shape)*(np.ones(eos_mask.shape)-eos_mask)
        # res=np.concatenate((res,target_seq),axis=1)
        # option 2
        for i in range(self.args.max_summary_len-1):
            output_tokens, *hc = self.decoder_model.predict([res]+e)
            sampled_token_index = np.argmax(output_tokens,axis =-1)
            # sampled_token = self.index2word_sum[sampled_token_index]
            # res = np.expand_dims(np.squeeze(sampled_token_index), axis=1)
            res=np.concatenate((np.expand_dims(target_seq),res),axis=1)
            # e = hc
        return res

    def id_seq2text_summary(self,seq):
        res=[]
        pad_id=self.word2index_sum['<PAD>']
        sos_id=self.word2index_sum['<SOS>']
        eos_id=self.word2index_sum['<EOS>']
        for x in seq:
            summary=''
            for t in x:
                if t==pad_id or t==sos_id:
                    continue
                if t==eos_id:
                    break
                summary+=" "+self.index2word_sum[int(t)]
            res.append(summary)
            # res.append(" ".join([self.index2word_sum[int(i)] for i in x if i!=pad_id and i!=sos_id and i!=eos_id]))
        return res
     
    def test(self,option='best'):
        if option=='current':
            self.model.load_weights(os.path.join(self.args.model_save_dir,'current_model_weight'))
        elif option=='best':
            self.model.load_weights(os.path.join(self.args.best_model_save_dir,'best_model_weight'))
        else:
            raise ValueError('parameter option can only take value "best" or "current"')
        rouge=ROUGE()
        precision=0
        recall=0
        f1score=0            
        
        with open(os.path.join(self.args.best_model_save_dir,'test predict.txt'),'w') as f:
            f.write('')
        for _ in range(0,len(self.x_test),self.args.batch_size):
                pred=self.predict(self.x_test[_:_+self.args.batch_size])
                pred=self.id_seq2text_summary(pred)
                score=rouge.rouge(self.test_summary[_:_+self.args.batch_size],pred,func='all')
                with open(os.path.join(self.args.best_model_save_dir,'test predict.txt'),'a') as f:
                    for txt,pre,rec,f1 in zip(pred,score['precision'],score['recall'],score['f1'] ):
                        f.write(json.dumps({'predict':txt,'precision':pre,'recall':rec,'f1':f1}))
                        f.write('\n')
                precision+=sum(score['precision'])
                recall+=sum(score['recall'])
                f1score+= sum(score['f1'])
        res={'precision':precision/len(self.x_test),'recall':recall/len(self.x_test),'f1':f1score/len(self.x_test)}
        print(f'ROUGE-1:{res}')
        with open(os.path.join(self.args.best_model_save_dir,f'rouge_score_test_{option}.json'),'w') as f:
                f.write(json.dumps(res))
                

    def summary(self,text):
        seq=text_to_word_sequence(text)
        seq=[self.word2index_text[i] for i in seq if i in self.word2index_text else self.word2index_text['<OOV>']]
        input_seq=pad_sequences([seq,seq],maxlen=self.args.max_text_len,padding='post')
        self.model.load_weights(os.path.join(self.args.best_model_save_dir,'best_model_weight'))
        output=self.predict(input_seq)
        output=self.id_seq2text_summary(output)
        return output[0]