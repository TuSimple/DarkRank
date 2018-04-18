short=128
qual=90

cd data/Market-list

im2rec train-even.lst / train-even.rec resize=$short quality=$qual
im2rec train-rand.lst / train-rand.rec resize=$short quality=$qual
im2rec test.lst / test.rec resize=$short quality=$qual
im2rec query.lst / query.rec resize=$short quality=$qual
im2rec gt-query.lst / query.rec resize=$short quality=$qual

