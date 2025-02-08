import pickle
from dataset import STRIDE
map_dir ="/home/stannis/Online-Action-Prediction-from-Scene-Graphs/processed10_d3"

# mapping_b[mode][idx] = (subject, task, take, frame_idx, actions)
# mapping_g[mode]["sample"][global_idx] = (subject, task, take, frame_idx_start, frame_idx_end, is_mirrored)
# path_mapping[mode][local_idx] = (data_path, global_idx)
def check_mappings(mapping_b, mapping_g, path_mapping, fold):
    for mode in ["train", "val", "test"]:
        assert len(mapping_g[mode]["sample"]) == len(path_mapping[mode]), f"Err: {len(mapping_g[mode]['sample'])} != {len(path_mapping[mode])}"
        assert 2*len(mapping_b[mode]) == len(path_mapping[mode]), f"Err: {2*len(mapping_b[mode])} != {len(path_mapping[mode])}" # FIXME: why is this not true?

        for local_idx, (data_path, global_idx) in path_mapping[mode].items():
            
            data_info1 = mapping_g[mode]["sample"][global_idx]
            
            if data_info1[-1] == "M":
                continue
            
            assert local_idx % 2 == 0, "local_idx is not even"
            
            idx = local_idx//2
            
            data_info2 = mapping_b[mode][idx]

            assert data_info1[0] == data_info2[0], f"Err: {data_info1[0]} != {data_info2[0]}" # subject 
            assert data_info1[1] == data_info2[1], f"Err: {data_info1[1]} != {data_info2[1]}" # task
            assert data_info1[2] == data_info2[2], f"Err: {data_info1[2]} != {data_info2[2]}" # take
            assert data_info1[3] == data_info2[3]//STRIDE, f"Err: {data_info1[3]} != {data_info2[3]}" # frame_idx_start
        
        print(f"{fold}-{mode} is ok")




# for fold in range(1, 7):
#     with open(f"data/sample_dict_fold{fold}.pickle", "rb") as f:
#         mapping_b = pickle.load(f)

#     with open(f"{map_dir}/dataset_mapping_fold{fold}.pickle", "rb") as f:
#         mapping_g = pickle.load(f)

#     with open(f"{map_dir}/path_mapping_fold{fold}.pickle", "rb") as f:
#         path_mapping = pickle.load(f)

#     check_mappings(mapping_b, mapping_g, path_mapping, fold)


