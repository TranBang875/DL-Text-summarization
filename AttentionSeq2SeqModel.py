from modelBase import Seq2SeqModel
from tensorflow.keras.layers import Input, LSTM,GRU, Embedding, Dense, Concatenate, TimeDistributed,Attention
from tensorflow.keras.models import Model
import numpy as np
class AttentionSeq2SeqModel(Seq2SeqModel):
    def __init__(self, args):
        super().__init__(args)
    
    def bulid_model(self):
                # model
        encoder_inputs = Input(shape=(self.args.max_text_len,))
        encoder_outputs =  Embedding(self.x_vocab_size, self.args.embedding_dim,trainable=True,mask_zero=True)(encoder_inputs)
        
        for i in range(self.args.encode_layer):
            if self.args.rnn_cell=='lstm':
                encoder_layer = LSTM(self.args.latent_dim,return_sequences=True,return_state=True,dropout=self.args.dropout_rate,recurrent_dropout=self.args.recurrent_dropout_rate)
            elif self.args.rnn_cell=='gru':
                encoder_layer = GRU(self.args.latent_dim,return_sequences=True,return_state=True,dropout=self.args.dropout_rate,recurrent_dropout=self.args.recurrent_dropout_rate)
            else:
                raise Exception("rnn cell is only lstm or gru")
            encoder_outputs, *encoder_state = encoder_layer(encoder_outputs)
        
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
        
        
        if self.args.attention_model=='dot':
            attention=Attention(score_mode='dot')
            attention_encoder=attention([decoder_outputs,encoder_outputs])
        elif self.args.attention_model=='general':
            attention=Attention(score_mode='dot')
            W=TimeDistributed(Dense(self.args.latent_dim))
            attention_encoder=attention([decoder_outputs,encoder_outputs,W(encoder_outputs)])
        elif self.args.attention_model=='concat':
            attention=Attention(score_mode='concat')
            W=TimeDistributed(Dense(self.args.latent_dim))
            attention_encoder=attention([W(decoder_outputs),encoder_outputs,W(encoder_outputs)])
        else:
            raise ValueError(
                f"Received: score_mode={self.attention_model}. Acceptable values are: [dot, general , concat]"   
            )
        concatenate=Concatenate()
        concat_output=concatenate([decoder_outputs,attention_encoder])
        decoder_dense =  TimeDistributed(Dense(self.y_vocab_size, activation='softmax'))
        decoder_outputs = decoder_dense(concat_output) 
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.model.compile(optimizer=self.args.optimizer, loss=self.args.loss)

        #Encoder model
        self.encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs]+encoder_state)
        # Decoder model
        decoder_state_input = [Input(shape=(self.args.latent_dim,))]
        if self.args.rnn_cell=='lstm':
            c = Input(shape=(self.args.latent_dim,))
            decoder_state_input.append(c)
        decoder_hidden_state_input = Input(shape=(self.args.max_text_len,self.args.latent_dim))
        decoder_outputs2= dec_emb_layer(decoder_inputs) 

        for layer in list_decoder:
            decoder_outputs2, *state = layer(decoder_outputs2, initial_state=decoder_state_input)
        if self.args.attention_model=='dot':
            _attention=attention([decoder_outputs2,decoder_hidden_state_input])
        elif self.args.attention_model=='general':
            _attention=attention([decoder_outputs2,decoder_hidden_state_input,W(decoder_hidden_state_input)])
        elif self.args.attention_model=='concat':
            _attention=attention([W(decoder_outputs2),decoder_hidden_state_input,W(decoder_hidden_state_input)])
        concat_output2=concatenate([decoder_outputs2,_attention])
        decoder_outputs2 = decoder_dense(concat_output2) 
        self.decoder_model = Model(
            [decoder_inputs,decoder_hidden_state_input] + decoder_state_input,
            [decoder_outputs2]+state)

        # # model
        # encoder_inputs = Input(shape=(self.args.max_text_len,))
        # enc_emb =  Embedding(self.x_vocab_size, self.args.embedding_dim,trainable=True,mask_zero=True)(encoder_inputs)
        # encoder_lstm1 = LSTM(self.args.latent_dim,return_sequences=True,return_state=True,dropout=self.args.dropout_rate,recurrent_dropout=self.args.recurrent_dropout_rate)
        # encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)
        # encoder_lstm2 = LSTM(self.args.latent_dim,return_sequences=True,return_state=True,dropout=self.args.dropout_rate,recurrent_dropout=self.args.recurrent_dropout_rate)
        # encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)
        # encoder_lstm3=LSTM(self.args.latent_dim, return_state=True, return_sequences=True,dropout=self.args.dropout_rate,recurrent_dropout=self.args.recurrent_dropout_rate)
        # encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)
        # decoder_inputs = Input(shape=(None,))
        # dec_emb_layer = Embedding(self.y_vocab_size, self.args.embedding_dim,trainable=True,mask_zero=True)
        # dec_emb = dec_emb_layer(decoder_inputs)
        # decoder_lstm = LSTM(self.args.latent_dim, return_sequences=True, return_state=True,dropout=self.args.dropout_rate,recurrent_dropout=self.args.recurrent_dropout_rate)
        # decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])
        # attention=Attention()
        # if self.args.attention_model=='dot':
        #     attention_encoder=attention([decoder_outputs,encoder_outputs])
        # elif self.args.attention_model=='general':
        #     W=TimeDistributed(Dense(self.args.latent_dim))
        #     attention_encoder=attention([decoder_outputs,encoder_outputs,W(encoder_outputs)])
        # elif self.args.attention_model=='concat':
        #     W=TimeDistributed(Dense(self.args.latent_dim))
        #     attention_encoder=attention([W(decoder_outputs),encoder_outputs,W(encoder_outputs)])
        # else:
        #     raise ValueError(
        #         f"Received: score_mode={self.attention_model}. Acceptable values are: [dot, general , concat]"   
        #     )
        # concatenate=Concatenate()
        # concat_output=concatenate([decoder_outputs,attention_encoder])
        # decoder_dense =  TimeDistributed(Dense(self.y_vocab_size, activation='softmax'))
        # decoder_outputs = decoder_dense(concat_output) 
        # self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        # self.model.compile(optimizer=self.args.optimizer, loss=self.args.loss)

        # #Encoder model
        # self.encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])
        # # Decoder model
        # decoder_state_input_h = Input(shape=(self.args.latent_dim,))
        # decoder_state_input_c = Input(shape=(self.args.latent_dim,))
        # decoder_hidden_state_input = Input(shape=(self.args.max_text_len,self.args.latent_dim))
        # dec_emb2= dec_emb_layer(decoder_inputs) 
        # decoder_outputs2, state_decoder_h, state_decoder_c = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])
        # if self.args.attention_model=='dot':
        #     _attention=attention([decoder_outputs2,decoder_hidden_state_input])
        # elif self.args.attention_model=='general':
        #     _attention=attention([decoder_outputs2,decoder_hidden_state_input,W(decoder_hidden_state_input)])
        # elif self.args.attention_model=='concat':
        #     _attention=attention([W(decoder_outputs2),decoder_hidden_state_input,W(decoder_hidden_state_input)])
        # concat_output2=concatenate([decoder_outputs2,_attention])
        # decoder_outputs2 = decoder_dense(concat_output2) 
        # self.decoder_model = Model(
        #     [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
        #     [decoder_outputs2,state_decoder_h, state_decoder_c])

    def predict(self,input_seq):
        e_out, *e = self.encoder_model.predict(input_seq)

        target_seq = np.ones((input_seq.shape[0],1))*self.word2index_sum['<SOS>']
        pad_id=self.word2index_sum['<PAD>']
        eos_id=self.word2index_sum['<EOS>']
        res=np.ones((input_seq.shape[0],1))*self.word2index_sum['<SOS>']
        # stop_condition = False
        # decoded_sentence = ''
        for i in range(self.args.max_summary_len-2):
            output_tokens, *hc = self.decoder_model.predict([target_seq,e_out]+e)
            sampled_token_index = np.argmax(output_tokens,axis =-1)
            # sampled_token = self.index2word_sum[sampled_token_index]
            target_seq = np.expand_dims(np.squeeze(sampled_token_index), axis=1)
            res=np.concatenate((res,target_seq),axis=1)
            e = hc
        not_eos_seq=target_seq-np.ones(target_seq.shape)*eos_id
        not_eos_seq=np.array(not_eos_seq,dtype=bool)
        not_pad_seq=target_seq-np.ones(target_seq.shape)*pad_id
        not_pad_seq=np.array(not_pad_seq,dtype=bool)        
        eos_mask=not_pad_seq*not_eos_seq
        target_seq=np.ones(eos_mask.shape)*eos_id*eos_mask+pad_id*np.ones(eos_mask.shape)*(np.ones(eos_mask.shape)-eos_mask)
        res=np.concatenate((res,target_seq),axis=1)
        return res        
