print("Running")
ofi_level = self.create_ofi(level, interval = interval)
if multi_level_ofi is None:
    print("No Interval. Merging without interval.")
    multi_level_ofi = ofi_level
else:
    multi_level_ofi = pd.merge(multi_level_ofi, ofi_level, on='ts_event', how='outer')