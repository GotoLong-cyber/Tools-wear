from data_provider.data_loader import UnivariateDatasetBenchmark, MultivariateDatasetBenchmark, Global_Temp, Global_Wind, Dataset_ERA5_Pretrain, Dataset_ERA5_Pretrain_Test, UTSD, UTSD_Npy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from data_provider.data_loader import PHM_MergedMultivariateNpy

data_dict = {
    'UnivariateDatasetBenchmark': UnivariateDatasetBenchmark,
    'MultivariateDatasetBenchmark': MultivariateDatasetBenchmark,
    'Global_Temp': Global_Temp,
    'Global_Wind': Global_Wind,
    'Era5_Pretrain': Dataset_ERA5_Pretrain,
    'Era5_Pretrain_Test': Dataset_ERA5_Pretrain_Test,
    'PHM_MergedMultivariateNpy': PHM_MergedMultivariateNpy,
    'Utsd': UTSD,
    'Utsd_Npy': UTSD_Npy
}


def data_provider(args, flag):
    Data = data_dict[args.data]

    if flag in ['test', 'val']:
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size
    use_persistent_workers = bool(getattr(args, "num_workers", 0) > 0)

    size = [args.seq_len, args.input_token_len, args.output_token_len] if flag in ['train', 'val'] \
        else [args.test_seq_len, args.input_token_len, args.test_pred_len]

    dataset_kwargs = dict(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=size,
        nonautoregressive=args.nonautoregressive,
        test_flag=args.test_flag,
        subset_rand_ratio=args.subset_rand_ratio,
    )

    if args.data == 'PHM_MergedMultivariateNpy':
        dataset_kwargs.update(
            keep_features_path=getattr(args, "keep_features_path", ""),
            train_runs=getattr(args, "train_runs", "c1,c4"),
            test_runs=getattr(args, "test_runs", "c6"),
            split_ratio=getattr(args, "split_ratio", 0.8),
            time_gap=getattr(args, "time_gap", 0),
            wear_agg=getattr(args, "wear_agg", "max"),
            mask_future_features_in_y=getattr(args, "mask_future_features_in_y", False),
        )

    data_set = Data(**dataset_kwargs)

    print(flag, len(data_set))

    if args.ddp:
        train_datasampler = DistributedSampler(data_set, shuffle=shuffle_flag)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            sampler=train_datasampler,
            num_workers=args.num_workers,
            persistent_workers=use_persistent_workers,
            pin_memory=True,
            drop_last=drop_last,
        )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            persistent_workers=use_persistent_workers,
            pin_memory=True,
            drop_last=drop_last
        )
    return data_set, data_loader
