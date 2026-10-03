import pyarrow.parquet as pq, time
pf=pq.ParquetFile('datasets/valid.parquet')
t=time.time(); x=pf.read_row_group(5); print('full rg', time.time()-t)
t=time.time(); x=pf.read_row_group(6,columns=['t0','t1','is_scored']); print('3 cols', time.time()-t)
pf=pq.ParquetFile('datasets/train.parquet')
t=time.time(); x=pf.read_row_group(5); print('train full rg', time.time()-t)
