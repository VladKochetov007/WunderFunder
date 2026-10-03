import pyarrow.parquet as pq
for f in ["datasets/train.parquet","datasets/valid.parquet","datasets/valid_mask.parquet"]:
    pf = pq.ParquetFile(f)
    md = pf.metadata
    print(f, "row_groups", md.num_row_groups, "rows", md.num_rows, "cols", md.num_columns)
    print(pf.schema_arrow)
    rg = md.row_group(0)
    print("rg0 rows", rg.num_rows, "bytes", rg.total_byte_size)
