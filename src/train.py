from sys import argv, stdout
from random import random, seed, shuffle
from functools import wraps
from time import gmtime, strftime, localtime, time
import subprocess
import pdb
sysargv = [a for a in argv]
import dynet as dy
seed(1)

from data import read_dataset, UNK, EOS, NONE, WF, LEMMA, MSD
VERBOSE=0
EARLY_STOPPING = True
LSTM_NUM_OF_LAYERS = 1
EMBEDDINGS_SIZE = 100
STATE_SIZE = 100
ATTENTION_SIZE = 100
WFDROPOUT=0.1
LSTMDROPOUT=0.3
# Every epoch, we train on a subset of examples from the train set,
# namely, 30% of them randomly sampled.
SAMPLETRAIN=1
start_time = time()

def iscandidatemsd(msd):
    """ We only consider nouns, verbs and adjectives. """
    return msd.split(';')[0] in ['N','V','ADJ']

def init_model():
    global model
    model = dy.Model()

def init_monolingual_params(wf2id,lemma2id,char2id,msd2id,languages):
    global monolingual_params
    monolingual_params = {}
    for lang in languages:
        monolingual_params[lang] = _init_monolingual_params(wf2id[lang],lemma2id[lang],char2id[lang],msd2id[lang])

def _init_monolingual_params(wf2id,lemma2id,char2id,msd2id):

    num_embedded_context_items=8
    enc_fwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, 2*STATE_SIZE+2*EMBEDDINGS_SIZE, STATE_SIZE, model)
    enc_bwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, 2*STATE_SIZE+2*EMBEDDINGS_SIZE, STATE_SIZE, model)

    dec_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, STATE_SIZE*2+EMBEDDINGS_SIZE, STATE_SIZE, model)

    character_lookup = model.add_lookup_parameters((len(char2id), EMBEDDINGS_SIZE))
    word_lookup = model.add_lookup_parameters((len(wf2id), EMBEDDINGS_SIZE))
    lemma_lookup = model.add_lookup_parameters((len(lemma2id), EMBEDDINGS_SIZE))
    msd_lookup = model.add_lookup_parameters((len(msd2id), EMBEDDINGS_SIZE))

    attention_w1 = model.add_parameters( (ATTENTION_SIZE, STATE_SIZE*2))
    attention_w2 = model.add_parameters( (ATTENTION_SIZE, STATE_SIZE*LSTM_NUM_OF_LAYERS*2))
    attention_v = model.add_parameters( (1, ATTENTION_SIZE))
    decoder_w = model.add_parameters( (len(char2id), STATE_SIZE))
    decoder_b = model.add_parameters( (len(char2id)))
    output_lookup = model.add_lookup_parameters((len(char2id), EMBEDDINGS_SIZE))#TO DO use the same lookup param for input and output

    context_fwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, 3*EMBEDDINGS_SIZE, STATE_SIZE, model)
    context_bwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, 3*EMBEDDINGS_SIZE, STATE_SIZE, model)

    return enc_fwd_lstm, enc_bwd_lstm, dec_lstm, character_lookup,\
        word_lookup, lemma_lookup, msd_lookup, attention_w1, attention_w2, \
        attention_v, decoder_w, decoder_b, output_lookup, context_fwd_lstm, context_bwd_lstm

def init_shared_params(msd2id_split):
    global dec_msd_lstm, decoder_msd_w, decoder_msd_b, output_msd_lookup

    dec_msd_lstm =  dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, 2*STATE_SIZE+2*EMBEDDINGS_SIZE, STATE_SIZE, model)# dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, STATE_SIZE*2+EMBEDDINGS_SIZE, STATE_SIZE, model)
    decoder_msd_w = model.add_parameters( (len(msd2id_split), STATE_SIZE))
    decoder_msd_b = model.add_parameters( (len(msd2id_split)))
    output_msd_lookup = model.add_lookup_parameters((len(msd2id_split), EMBEDDINGS_SIZE))#TO DO use the same lookup param for input and output


def embed(lemma,context):
    """ Get word embedding and character based embedding for the input
        lemma. Concatenate the embeddings with a context representation. """
    lemma = [EOS] + list(lemma) + [EOS]
    lemma = [c if c in char2id else UNK for c in lemma]
    lemma = [char2id[c] for c in lemma]

    global character_lookup

    return [dy.concatenate([character_lookup[c], context])
            for c in lemma]

def run_lstm(init_state, input_vecs):
    s = init_state
    out_vectors = []
    for vector in input_vecs:
        s = s.add_input(vector)
        out_vector = s.output()
        out_vectors.append(out_vector)
    return out_vectors


def encode(embedded):
    embedded_rev = list(reversed(embedded))

    fwd_vectors = run_lstm(enc_fwd_lstm.initial_state(), embedded)
    bwd_vectors = run_lstm(enc_bwd_lstm.initial_state(), embedded_rev)
    bwd_vectors = list(reversed(bwd_vectors))
    vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]
    first_and_last = [fwd_vectors[-1],bwd_vectors[0]]
    return vectors, first_and_last


def attend(input_mat, state, w1dt):
    global attention_w2
    global attention_v
    w2 =attention_w2
    v =attention_v

    # input_mat: (encoder_state x seqlen) => input vecs concatenated as cols
    # w1dt: (attdim x seqlen)
    # w2dt: (attdim x attdim)
    w2dt = w2*dy.concatenate(list(state.s()))
    # att_weights: (seqlen,) row vector
    unnormalized = dy.transpose(v * dy.tanh(dy.colwise_add(w1dt, w2dt)))
    att_weights = dy.softmax(unnormalized)
    # context: (encoder_state)
    context = input_mat * att_weights
    return context

def decode(vectors, output, decode_char=True):#if char --> decode into characters to produce tgt form; else decode into msd's to produce tag sequence
    #if not decode_char: pdb.set_trace()
    output = [EOS] + list(output) + [EOS]
    if decode_char:
        x2id = char2id
        output_x_lookup = output_lookup
        w = decoder_w
        b = decoder_b
        w1 =attention_w1
        x_lstm = dec_lstm
        input_mat = dy.concatenate_cols(vectors)
    else:
        x2id = msd2id_split
        output_x_lookup = output_msd_lookup
        w = decoder_msd_w
        b = decoder_msd_b
        x_lstm = dec_msd_lstm
        input_mat = vectors#dy.concatenate(vectors)

    output = [x2id[c] for c in output]

    w1dt = None

    last_output_embeddings = output_x_lookup[x2id[EOS]]
    if decode_char:
        s = x_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE*2), last_output_embeddings]))
    else:
        s = x_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE*2+EMBEDDINGS_SIZE), last_output_embeddings]))
    loss = []

    for char in output:

        # w1dt can be computed and cached once for the entire decoding phase
        if decode_char:
            w1dt = w1dt or w1 * input_mat
            vector = dy.concatenate([attend(input_mat, s, w1dt), last_output_embeddings])
        else:
            vector = dy.concatenate([input_mat, last_output_embeddings])
        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector)
        last_output_embeddings = output_x_lookup[char]
        loss.append(-dy.log(dy.pick(probs, char)))
    loss = dy.esum(loss)
    return loss


def generate(i, s, id2x, generate_char=True):
    """ Generate a word form for the lemma at position i in sentence s. """
    context = get_context(i,s)
    embedded = embed(s[i][LEMMA],context)
    encoded, first_and_last = encode(embedded)

    in_seq = s[i][LEMMA]
    if generate_char:
        x2id = char2id
        output_x_lookup = output_lookup
        w = decoder_w
        b = decoder_b
        w1 =attention_w1
        x_lstm = dec_lstm
        input_mat = dy.concatenate_cols(encoded)
    else:
        x2id = msd2id_split
        output_x_lookup = output_msd_lookup
        w = decoder_msd_w
        b = decoder_msd_b
        x_lstm = dec_msd_lstm
        input_mat = context #dy.concatenate(first_and_last)

    w1dt = None

    last_output_embeddings = output_x_lookup[x2id[EOS]]
    if generate_char:
        s = x_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE*2), last_output_embeddings]))
    else:
        s = x_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE*2+EMBEDDINGS_SIZE), last_output_embeddings]))

    out = []
    count_EOS = 0
    for i in range(len(in_seq)*2):
        if count_EOS == 2: break
        # w1dt can be computed and cached once for the entire decoding phase
        if generate_char:
            w1dt = w1dt or w1 * input_mat
            vector = dy.concatenate([attend(input_mat, s, w1dt), last_output_embeddings])
        else:
            vector = dy.concatenate([input_mat, last_output_embeddings])

        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector).vec_value()
        next_char = probs.index(max(probs))
        #if not generate_char: import pdb; pdb.set_trace()
        last_output_embeddings = output_x_lookup[next_char]
        if id2x[next_char] == EOS:
            count_EOS += 1
            continue

        out.append(id2x[next_char])

    if generate_char: out=''.join(out)
    else: out=';'.join(out)
    return out

def dropitem(item,item2id,training):
    return item2id[UNK if not item in item2id
                   or training and random() < WFDROPOUT else
                   item]

def embed_context(prevword,prevlemma,prevmsd,lemma,
                  nextword,nextlemma,nextmsd):
    """ Emebed context elements. """

    prev_embedded, next_embedded = [],[]
    for w,l,m in zip(prevword,prevlemma,prevmsd):
        prev_embedded.append(dy.concatenate([word_lookup[w],lemma_lookup[l],msd_lookup[m]]))
    for w,l,m in zip(nextword,nextlemma,nextmsd):
        next_embedded.append(dy.concatenate([word_lookup[w],lemma_lookup[l],msd_lookup[m]]))
    lemma_embedded = lemma_lookup[lemma]
    return encode_context(prev_embedded,next_embedded,lemma_embedded)

def encode_context(prev_embedded, next_embedded, lemma_embedded):

    fwd_vectors = run_lstm(context_fwd_lstm.initial_state(), prev_embedded)
    bwd_vectors = run_lstm(context_bwd_lstm.initial_state(), next_embedded)

    return dy.concatenate([fwd_vectors[-1],bwd_vectors[-1],lemma_embedded])

def get_context(i,s,training=0):
    """ Embed context words, lemmas and MSDs.

        The context of a lemma consists of the previous and following
        word forms, lemmas and MSDs as well as the MSD for the lemma
        in question.
    """
    prevword, nextword, prevmsd, nextmsd, prevlemma, nextlemma = [],[],[],[],[],[]
    for j in range(0,i+1):
        prevword.append(dropitem(s[j-1][WF] if j > 0 else EOS,wf2id,training))
        prevlemma.append(dropitem(s[j-1][LEMMA] if j > 0 else EOS, lemma2id,training))
        prevmsd.append(dropitem(s[j-1][MSD] if j > 0 else EOS, msd2id,training))
    for j in range(i,len(s)):
        nextword.append(dropitem(s[j+1][WF] if j + 1 < len(s) else EOS,wf2id,training))
        nextlemma.append(dropitem(s[j+1][LEMMA] if j + 1 < len(s) else EOS,lemma2id,training))
        nextmsd.append(dropitem(s[j+1][MSD] if j + 1 < len(s) else EOS,msd2id,training))

    lemma = s[i][LEMMA] if i > 0 else EOS
    lemma = dropitem(lemma,lemma2id,training)

    return embed_context(prevword, prevlemma, prevmsd, lemma, nextword, nextlemma, nextmsd)

def get_loss(i, s, validation=False):
    dy.renew_cg()
    enc_fwd_lstm.set_dropout(LSTMDROPOUT)
    enc_bwd_lstm.set_dropout(LSTMDROPOUT)
    dec_lstm.set_dropout(LSTMDROPOUT)

    context = get_context(i,s,training=1)
    embedded = embed(s[i][LEMMA], context)
    encoded, first_and_last = encode(embedded)
    loss =  decode(encoded, s[i][WF])

    enc_fwd_lstm.set_dropout(0)
    enc_bwd_lstm.set_dropout(0)
    dec_lstm.set_dropout(0)

    if s[i][MSD] is not NONE and iscandidatemsd(s[i][MSD]) and not validation: #this ensures that MSD is not predicted in track 2
        #loss_msd = decode(first_and_last, s[i][MSD].split(';'), False)
        loss_msd = decode(context, s[i][MSD].split(';'), False)
        return dy.esum([0.7*loss,0.3*loss_msd])
    else: return loss

def memolrec(func):
    """Memoizer for Levenshtein."""
    cache = {}
    @wraps(func)
    def wrap(sp, tp, sr, tr, cost):
        if (sr,tr) not in cache:
            res = func(sp, tp, sr, tr, cost)
            cache[(sr,tr)] = (res[0][len(sp):], res[1][len(tp):], res[4] - cost)
        return sp + cache[(sr,tr)][0], tp + cache[(sr,tr)][1], '', '', cost + cache[(sr,tr)][2]
    return wrap

def levenshtein(s, t, inscost = 1.0, delcost = 1.0, substcost = 1.0):
    """Recursive implementation of Levenshtein, with alignments returned.
       Courtesy of Mans Hulden. """
    @memolrec
    def lrec(spast, tpast, srem, trem, cost):
        if len(srem) == 0:
            return spast + len(trem) * '_', tpast + trem, '', '', cost + len(trem)
        if len(trem) == 0:
            return spast + srem, tpast + len(srem) * '_', '', '', cost + len(srem)

        addcost = 0
        if srem[0] != trem[0]:
            addcost = substcost

        return min((lrec(spast + srem[0], tpast + trem[0], srem[1:], trem[1:], cost + addcost),
                    lrec(spast + '_', tpast + trem[0], srem, trem[1:], cost + inscost),
                    lrec(spast + srem[0], tpast + '_', srem[1:], trem, cost + delcost)),
                   key = lambda x: x[4])

    answer = lrec('', '', s, t, 0)
    return answer[0],answer[1],answer[4]

def eval(devdata, val_loss, best_epoch, generating=1, outf=None, out_scores=None):
    if VERBOSE: print('Predicting development data with best model...')
    model.populate('{}/model'.format(exp_path))
    input, gold = devdata
    corr = 0.0
    lev=0.0
    tot = 0.0

    for n,s in enumerate(input):
        for i,fields in enumerate(s):
            dy.renew_cg()
            wf, lemma, msd = fields
            if gold[n][i][MSD] == NONE and lemma != NONE:
                if generating:
                    wf = generate(i,s,id2char)
                if wf == gold[n][i][WF]:
                    corr += 1
                lev += levenshtein(wf,gold[n][i][WF])[2]
                tot += 1
            if outf:
                outf.write('%s\n' % '\t'.join([wf,lemma,msd]))
        if outf:
            outf.write('\n')

    to_print = ''
    total_time = time() - start_time
    if out_scores is not None:
        for lab, item in zip(['Dev set accuracy','Lev dist','Val loss' ,'Epoch', 'Time'],[corr/tot*100, lev/tot, val_loss, best_epoch, total_time]):
            out_scores.write('{}: {}\n'.format(lab,str(item)))
            to_print+='%.2f\t' % item
        #to_print+='\n'
    #print(to_print)

    return (0,0, 0) if tot == 0 else (corr/tot, lev/tot, to_print)

def eval_msd(devdata,generating=1, outf=None, out_scores=None):
    if VERBOSE: print('Predicting development data with best model...')
    model.populate('{}/model'.format(exp_path))
    input, gold = devdata
    corr = 0.0
    lev=0.0
    tot = 0.0
    #pdb.set_trace()
    for n,s in enumerate(input):
        for i,fields in enumerate(s):
            dy.renew_cg()
            wf, lemma, msd = fields
            if iscandidatemsd(gold[n][i][MSD]):
                if generating:
                    msd = generate(i,s,id2msd_split, generate_char=False)
                if msd == gold[n][i][MSD]:
                    corr += 1
                lev += levenshtein(msd,gold[n][i][MSD])[2]
                tot += 1
                msd = gold[n][i][MSD]#for all words that are not actual targets, write out their real label
            if gold[n][i][MSD] is NONE and generating:
                msd = generate(i,s,id2msd_split,generate_char=False)
            if outf:
                outf.write('%s\n' % '\t'.join([wf,lemma,msd]))
        if outf:
            outf.write('\n')

    if out_scores is not None:
        for lab, item in zip(['Dev set accuracy MSD','Lev dist MSD'],[corr/tot*100, lev/tot]):
            out_scores.write('{}: {}\n'.format(lab,str(item)))
    to_print = corr/tot*100
    return (0,0,0) if tot == 0 else (corr / tot, lev/tot, to_print)

def set_language(lang):
    global enc_fwd_lstm, enc_bwd_lstm, dec_lstm, character_lookup,\
    word_lookup, lemma_lookup, msd_lookup, attention_w1, attention_w2, \
    attention_v, decoder_w, decoder_b, output_lookup, context_fwd_lstm, context_bwd_lstm, \
    wf2id,lemma2id,char2id,id2char,msd2id

    enc_fwd_lstm, enc_bwd_lstm, dec_lstm, character_lookup,\
    word_lookup, lemma_lookup, msd_lookup, attention_w1, attention_w2, \
    attention_v, decoder_w, decoder_b, output_lookup, context_fwd_lstm, context_bwd_lstm = monolingual_params[lang]

    wf2id,lemma2id,char2id,id2char,msd2id = wf2id_dict[lang],lemma2id_dict[lang],char2id_dict[lang],id2char_dict[lang],msd2id_dict[lang]

def predict(testdata,lang, outf):
    input = testdata
    set_language(lang)
    for n,s in enumerate(input):
        for i,fields in enumerate(s):
            dy.renew_cg()
            wf, lemma, msd = fields
            if msd == NONE and lemma != NONE:
                wf = generate(i,s,id2char,lang)
            outf.write('%s\n' % '\t'.join([wf,lemma,msd]))
        outf.write('\n')

def train(traindata,devdata,epochs=20,finetuning=False):
    trainer = dy.AdamTrainer(model)
    if finetuning:
        trainer.learning_rate=0.0001
    
    valdata = {lang: traindata[lang][int(0.7*len(traindata[lang])):int(0.8*len(traindata[lang]))] for lang in languages} #hold out 10% for validation loss
    traindata = {lang: traindata[lang][:int(0.7*len(traindata[lang]))]+traindata[lang][int(0.8*len(traindata[lang])):] for lang in languages} #use only the other 90% for training
    prev_best_loss = 10000
    best_epoch = 0
    n_no_improvement = 0
    shortest_subset = min([len(traindata[lang]) for lang in languages])
    for epoch in range(epochs):
        if VERBOSE: print("EPOCH %u" % (epoch + 1))
        for lang in languages: shuffle(traindata[lang])
        total_train_loss, multilingual_val_loss = 0, 0
        # train on 90% of data

        for n in range(shortest_subset):
            shuffle(languages)
            for lang in languages:
                set_language(lang)
                s = traindata[lang][n]
                for i,fields in enumerate(s):
                    wf, lemma, msd = fields
                    if VERBOSE: stdout.write("Example %u of %u\r" %
                                 (n+1,len(traindata[lang])))
                    if (iscandidatemsd(msd) or (msd == NONE and lemma != NONE))\
                       and random() < SAMPLETRAIN:
                        loss = get_loss(i, s)
                        loss.backward()
                        loss_value = loss.value()
                        total_train_loss += loss_value
                        trainer.update()

        # compute loss on the other 10% without backproping it
        for lang in languages:
            total_val_loss = 0
            set_language(lang)
            for n,s in enumerate(valdata[lang]):
                for i,fields in enumerate(s):
                    wf, lemma, msd = fields
                    if (iscandidatemsd(msd) or (msd == NONE and lemma != NONE))\
                       and random() < SAMPLETRAIN:
                        loss = get_loss(i, s, validation=True)
                        loss_value = loss.value()
                        total_val_loss += loss_value
            if VERBOSE:
                print("\nTraining loss per sentence: %.3f" % (total_train_loss/int(len(traindata[lang])*SAMPLETRAIN)))
                print("Validation loss per sentence: %.3f" % (total_val_loss/int(len(valdata[lang])*SAMPLETRAIN)))
                print("Example outputs:")

                for s in valdata[lang][2:4]:
                    for i,fields in enumerate(s):
                        wf, lemma, msd = fields
                        if (iscandidatemsd(msd) or (msd == NONE and lemma != NONE))\
                           and random() < SAMPLETRAIN:
                            if VERBOSE: print("INPUT:", s[i][LEMMA], "OUTPUT:",
                                 generate(i,s,id2char),
                                 "GOLD:",wf)
                            break
            multilingual_val_loss+=total_val_loss/len(valdata[lang])
        #save best model so far
        if multilingual_val_loss < prev_best_loss:
            if VERBOSE: print('+++++++New best dev loss++++++')
            model.save('{}/model'.format(exp_path))
            prev_best_loss = multilingual_val_loss
            best_epoch = epoch+1
            n_no_improvement = 0
        else: n_no_improvement +=1

        #early stopping
        if (n_no_improvement == 5 and EARLY_STOPPING) or epoch == epochs-1:
            if VERBOSE: print('Early stopping after 5 epochs of no improvement...') if n_no_improvement == 5 else print('Finished training...')
            for lang in languages:
                set_language(lang)
                devacc, devlev, to_print = eval((devdata[0][lang],devdata[1][lang]),
                     prev_best_loss,best_epoch,
                     generating=1,
                     outf=open("{}/{}_out".format(exp_path,lang),"w"),
                     out_scores=open("{}/{}_scores".format(exp_path,lang),"w"))
                _,_,to_print_msd = eval_msd((devdata[0][lang],devdata[1][lang]),
                     generating=1,
                     outf=open("{}/{}_out_msd".format(exp_path,lang),"w"),
                     out_scores=open("{}/{}_scores_msd".format(exp_path,lang),"w"))
                print(exp_name+' '+lang+'\t'+str(to_print)+str(to_print_msd)+'\n')
                if VERBOSE:
                    print("Development set accuracy for best %s model: %.2f" % (lang, 100*devacc))
                    print("Development set avg. Levenshtein %s distance: %.2f" % (lang,devlev))
                    print()
            break

if __name__=='__main__':

    exp_name = str(sysargv[5])
    exp_path = 'dumped/'+exp_name
    global wf2id_dict, lemma2id_dict, char2id_dict, id2char_dict, msd2id_dict, msd2id_split, id2msd_split,languages
    languages = str(sysargv[1]).split(',')
    traindata, devinputdata, devgolddata, wf2id_dict, lemma2id_dict, char2id_dict, msd2id_dict, id2char_dict, msd2id_split_monolingual, id2msd_split = {},{},{},{},{},{},{},{},{},{}

    for lang in languages:
        traindata[lang], wf2id_dict[lang], lemma2id_dict[lang], char2id_dict[lang], msd2id_dict[lang], \
            msd2id_split_monolingual[lang] = read_dataset(sysargv[2].format(lang))
        devinputdata[lang], _, _, _, _, _ = read_dataset(sysargv[3].format(lang))
        devgolddata[lang], _, _, _, _, _ = read_dataset(sysargv[4].format(lang))

        id2char_dict[lang] = {id:char for char,id in char2id_dict[lang].items()}

    all_msd_splits = []
    for i,lang in enumerate(languages):
        if i==0: all_msd_splits+=msd2id_split_monolingual[lang].keys()
        else:
            for m in msd2id_split_monolingual[lang].keys():
                if m not in all_msd_splits: all_msd_splits.append(m)

    msd2id_split = {i:j for j,i in enumerate(all_msd_splits)}
    id2msd_split =  {id:msd for msd,id in msd2id_split.items()}

    init_model()
    init_shared_params(msd2id_split)
    init_monolingual_params(wf2id_dict,lemma2id_dict,char2id_dict,msd2id_dict,languages)

    mode = str(sysargv[6])
    if mode == 'training':
        subprocess.Popen("mkdir %s" % exp_path, shell=True).wait()
        if VERBOSE: print('The experiment will be saved in {}/'.format(exp_path))
        train(traindata,[devinputdata,devgolddata],50)
    elif mode == 'finetuning':
        ori_exp_path = exp_path
        ori_exp_name = exp_name
        model.populate('{}/model'.format(ori_exp_path))
        finetune_languages = sysargv[7].strip().split(',')
        for lang in finetune_languages:
            exp_path = ori_exp_path+'_'+lang
            exp_name = ori_exp_name+'_'+lang
            languages=[lang]
            subprocess.Popen("mkdir %s" % exp_path, shell=True).wait()
            if VERBOSE: print('The experiment will be saved in {}/'.format(exp_path))
            train(traindata,[devinputdata,devgolddata],
                epochs=5, finetuning=True)
    elif mode == 'testing':
        test_language=sysargv[7]
        setting = sysargv[8]
        exp_path = 'dumped/'+exp_name#+'_'+lang
        model.populate('dumped/{}/model'.format(exp_name))
        testdata_path = 'testsets/{}-track1-covered'.format(test_language)
        testdata, _, _, _, _, _ = read_dataset(testdata_path)
        predict(testdata, test_language, open("{}/{}-{}-out".format(exp_path,test_language,setting),"w"))
