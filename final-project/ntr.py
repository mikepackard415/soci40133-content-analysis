import pandas as pd
import numpy as np

lda100 = pd.read_csv('../Data/Environmental-Discourse/Full-TMs/lda100_loadings.csv', index_col=0)
for i in range(100):
    lda100['topic_{}'.format(i)] = lda100['topic_{}'.format(i)].apply(lambda x: x + 0.000001 if x == 0 else x)


doc_topic = np.array(lda100[['topic_{}'.format(i) for i in range(100)]])

lda100['date'] = pd.to_datetime(lda100.date)
lda100['year'] = lda100.date.dt.year

first_day = lda100.date.min()
lda100['date_index'] = lda100.date.apply(lambda x: (x - first_day) / np.timedelta64(1, 'D'))

window = 365
novelties = []
transiences = []

def kld(a, b):
    return (a * np.log2(a/b)).sum()

for i in range(int(lda100.date_index.max())+1):
    current = doc_topic[lda100.date_index == i, :]
    past =    doc_topic[(lda100.date_index < i) & (i - lda100.date_index < window), :]
    future =  doc_topic[(lda100.date_index > i) & (lda100.date_index - i > window), :]
    
    for j in range(current.shape[0]):
        kld_before = []
        kld_after = []
        a = current[j, :]
        
        for k in range(past.shape[0]):
            b = past[k, :]
            kld_before.append(kld(a, b))
        
        for k in range(future.shape[0]):
            b = future[k, :]
            kld_after.append(kld(a, b))
            
        if kld_before == []:
            novelties.append(np.nan)
        else:
            novelties.append(np.mean(kld_before))
        if kld_after == []:
            transiences.append(np.nan)
        else:
            transiences.append(np.mean(kld_after))


ntr = lda100.copy()[['url', 'title', 'date', 'year']]
ntr['novelty'] = novelties
ntr['transience'] = transiences
ntr['resonance'] = ntr.novelty - ntr.transience

ntr.to_csv('../Data/Environmental-Discourse/Full-TMs/ntr365.csv')
ntr.to_pickle('../Data/Environmental-Discourse/Full-TMs/ntr365.pkl')
