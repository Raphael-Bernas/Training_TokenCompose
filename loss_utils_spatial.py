import torch
import torch.nn as nn
from torchvision import transforms
from copy import deepcopy
import numpy as np

SD14_TO_SD21_RATIO = 1.5

# get token index in text
def get_word_idx(text: str, tgt_word, tokenizer):

    tgt_word = tgt_word.lower()

    # ignore the first and last token
    encoded_text = tokenizer.encode(text)[1:-1]
    encoded_tgt_word = tokenizer.encode(tgt_word)[1:-1]

    # find the idx of target word in text
    first_token_idx = -1
    for i in range(len(encoded_text)):
        if encoded_text[i] == encoded_tgt_word[0]:

            if len(encoded_text) > 0:
                # check the following 
                following_match = True
                for j in range(1, len(encoded_tgt_word)):
                    if encoded_text[i + j] != encoded_tgt_word[j]:
                        following_match = False
                if not following_match:
                    continue
            # for a single encoded idx, just take it
            first_token_idx = i

            break

    assert first_token_idx != -1, "word not in text"

    # add 1 for sot token
    tgt_word_tokens_idx_ls = [i + 1 + first_token_idx for i in range(len(encoded_tgt_word))]

    # sanity check
    encoded_text = tokenizer.encode(text)

    decoded_token_ls = []

    for word_idx in tgt_word_tokens_idx_ls:
        text_decode = tokenizer.decode([encoded_text[word_idx]]).strip("#")
        decoded_token_ls.append(text_decode)

    decoded_tgt_word = "".join(decoded_token_ls)
    
    tgt_word_ls = tgt_word.split(" ")
    striped_tgt_word = "".join(tgt_word_ls).strip("#")

    assert decoded_tgt_word == striped_tgt_word, "decode_text != striped_tar_wd"

    return tgt_word_tokens_idx_ls

def get_spatial_loss(cross_attention_before, cross_attention_after, mask_before, mask_after, relation):
    _device_ = cross_attention_before.device
    epsilon = 1e-6
    # Apply the mask to set values to 0 where mask is False
    masked_values_before = cross_attention_before.masked_fill(~mask_before.bool(), 0)
    masked_values_after = cross_attention_after.masked_fill(~mask_after.bool(), 0)
    # max value before
    max_value_before, idx_before = masked_values_before.view(-1).max(0)
    # max value after
    max_value_after, idx_after = masked_values_after.view(-1).max(0)
    max_i_before, max_j_before = -1, -1

    for i in range(mask_before.size(1)):
        if mask_before[0][i].sum() < epsilon:
            max_i_before = i
            break
    for j in range(mask_before.size(2)):
        if mask_before[0][:, j].sum() < epsilon:
            max_j_before = j
            break
    map_values_total = masked_values_before - masked_values_after
    if relation == "left":
        corr_matrix = np.zeros(map_values_total.size(2))
        corr_matrix[:max_i_before] = -1.0
        corr_matrix[max_i_before:] = 1.0
        corr_matrix = torch.tensor(corr_matrix)
        corr_matrix = corr_matrix.to(_device_)
        spatial_loss = torch.sum(torch.abs(map_values_total * corr_matrix))
    elif relation == "right":
        corr_matrix = np.zeros(map_values_total.size(2))
        corr_matrix[:max_i_before] = 1.0
        corr_matrix[max_i_before:] = -1.0
        corr_matrix = torch.tensor(corr_matrix)
        corr_matrix = corr_matrix.to(_device_)
        spatial_loss = torch.sum(torch.abs(map_values_total * corr_matrix))
    elif relation == "top":
        corr_matrix = np.zeros(map_values_total.size(1))
        corr_matrix[:max_j_before] = 1.0
        corr_matrix[max_j_before:] = -1.0
        corr_matrix = torch.tensor(corr_matrix)
        corr_matrix = corr_matrix.to(_device_)
        spatial_loss = torch.sum(torch.abs(map_values_total.transpose(1, 2) * corr_matrix))
    elif relation == "bottom":
        corr_matrix = np.zeros(map_values_total.size(1))
        corr_matrix[:max_j_before] = -1.0
        corr_matrix[max_j_before:] = 1.0
        corr_matrix = torch.tensor(corr_matrix)
        corr_matrix = corr_matrix.to(_device_)
        spatial_loss = torch.sum(torch.abs(map_values_total.transpose(1, 2) * corr_matrix))
    else:
        raise ValueError("Invalid relation")
    spatial_loss = 1.0 - spatial_loss / (mask_before.sum() + mask_after.sum())
    return spatial_loss

# get attn loss by resolution
def get_grounding_loss_by_layer(_gt_seg_list, word_token_idx_ls, res, 
                                input_attn_map_ls, is_training_sd21, spatial_results):
    if is_training_sd21:
        # training with sd21, using resolution 768 = 512 * 1.5
        res = int(SD14_TO_SD21_RATIO * res)

    gt_seg_list = deepcopy(_gt_seg_list)

    # reszie gt seg map to the same size with attn map
    resize_transform = transforms.Resize((res, res))

    for i in range(len(gt_seg_list)):
        gt_seg_list[i] = resize_transform(gt_seg_list[i])
        gt_seg_list[i] = gt_seg_list[i].squeeze(0) # 1, 1, res, res => 1, res, res
        # add binary
        gt_seg_list[i] = (gt_seg_list[i] > 0.0).float()


    ################### token loss start ###################
    # Following code is adapted from
    # https://github.com/silent-chen/layout-guidance/blob/08b687470f911c7f57937012bdf55194836d693e/utils.py#L27
    token_loss = 0.0
    for attn_map in input_attn_map_ls:
        b, H, W, j = attn_map.shape
        for i in range(len(word_token_idx_ls)): # [[word1 token_idx1, word1 token_idx2, ...], [word2 token_idx1, word2 token_idx2, ...]]
            obj_loss = 0.0
            single_word_idx_ls = word_token_idx_ls[i] #[token_idx1, token_idx2, ...]
            mask = gt_seg_list[i]

            for obj_position in single_word_idx_ls:
                # ca map obj shape 8 * 16 * 16
                ca_map_obj = attn_map[:, :, :, obj_position].reshape(b, H, W)

                activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)

                obj_loss += (1.0 - torch.mean(activation_value)) ** 2

            token_loss += (obj_loss/len(single_word_idx_ls))

    # normalize with len words
    token_loss = token_loss / len(word_token_idx_ls)
    ################## token loss end ##########################

    ################## pixel loss start ######################
    # average cross attention map on different layers
    avg_attn_map_ls = []
    for i in range(len(input_attn_map_ls)):
        avg_attn_map_ls.append(
            input_attn_map_ls[i].reshape(-1, res, res, input_attn_map_ls[i].shape[-1]).mean(0)
        )
    avg_attn_map = torch.stack(avg_attn_map_ls, dim=0)
    avg_attn_map = avg_attn_map.sum(0) / avg_attn_map.shape[0]
    avg_attn_map = avg_attn_map.unsqueeze(0)

    bce_loss_func = nn.BCELoss()
    pixel_loss = 0.0

    for i in range(len(word_token_idx_ls)):
        word_cross_attn_ls = []
        for token_idx in word_token_idx_ls[i]:
            word_cross_attn_ls.append(
                avg_attn_map[..., token_idx]
            )
        word_cross_attn_ls = torch.stack(word_cross_attn_ls, dim=0).sum(dim=0)
        pixel_loss += bce_loss_func(word_cross_attn_ls, gt_seg_list[i])

    # average with len word_token_idx_ls
    pixel_loss = pixel_loss / len(word_token_idx_ls)
    ################## pixel loss end #########################

    ################### spatial loss start ###################
    # average cross attention map on different layers
    avg_attn_map_ls = []
    for i in range(len(input_attn_map_ls)):
        avg_attn_map_ls.append(
            input_attn_map_ls[i].reshape(-1, res, res, input_attn_map_ls[i].shape[-1]).mean(0)
        )
    avg_attn_map = torch.stack(avg_attn_map_ls, dim=0)
    avg_attn_map = avg_attn_map.sum(0) / avg_attn_map.shape[0]
    avg_attn_map = avg_attn_map.unsqueeze(0)
    spatial_loss = 0.0
    for i in range(len(spatial_results)//2):
        if len(spatial_results[list(spatial_results.keys())[2*i]]) != 0:
            for attn_map in input_attn_map_ls:
                b, H, W, j = attn_map.shape
                i_before = spatial_results[list(spatial_results.keys())[2*i]][-1] # last word of the words before
                i_after = spatial_results[list(spatial_results.keys())[2*i+1]][0] # first word of the words after
                mask_before = gt_seg_list[i_before]
                mask_after = gt_seg_list[i_after]
                word_cross_attn_ls_before = []
                for token_idx in word_token_idx_ls[i_before]:
                    word_cross_attn_ls_before.append(
                        avg_attn_map[..., token_idx]
                    )
                word_cross_attn_ls_after = []
                for token_idx in word_token_idx_ls[i_after]:
                    word_cross_attn_ls_after.append(
                        avg_attn_map[..., token_idx]
                    )
                word_cross_attn_ls_before = torch.stack(word_cross_attn_ls_before, dim=0).sum(dim=0)
                word_cross_attn_ls_after = torch.stack(word_cross_attn_ls_after, dim=0).sum(dim=0)
                spatial_loss += get_spatial_loss(word_cross_attn_ls_before, word_cross_attn_ls_after, mask_before, mask_after, list(spatial_results.keys())[2*i].split("1")[0])

                

    ################## spatial loss end ##########################

    return {
        "token_loss" : token_loss,
        "pixel_loss": pixel_loss,
        "spatial_loss": spatial_loss
    }

