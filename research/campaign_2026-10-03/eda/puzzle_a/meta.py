import pyarrow.parquet as pq
for f in ["datasets/train.parquet","datasets/valid.parquet","datasets/valid_mask.parquet"]:
    pf = pq.ParquetFile(f)
    md = pf.metadata
    print(f, "rows", md.num_rows, "rgs", md.num_row_groups, "cols", md.num_columns)
    print(pf.schema_arrow)
    print("kv meta keys:", list((md.metadata or {}).keys()))
    print("created_by:", md.created_by)
    rg = md.row_group(0)
    print("rg0 rows", rg.num_rows, "bytes", rg.total_byte_size)
    for i in range(min(rg.num_columns, 80)):
        c = rg.column(i)
        st = c.statistics
        print("  ", c.path_in_schema, c.physical_type, c.compression, st.min if st and st.has_min_max else None, st.max if st and st.has_min_max else None)
