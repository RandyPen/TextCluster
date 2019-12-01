# -*- coding: utf-8 -*-
import os
import argparse
from collections import defaultdict

from tqdm import tqdm

from utils.similar import jaccard
from utils.segmentor import Segmentor
from utils.utils import check_file, ensure_dir, clean_dir, sample_file, get_stop_words, line_counter


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, default='./data/infile', help='Directory of input file.')
    parser.add_argument('--output', type=str, default='./data/output', help='Directory to save output file.')
    parser.add_argument('--dict', type=str, default='./data/seg_dict', help='Directory of dict file.')
    parser.add_argument('--stop_words', type=str, default='./data/stop_words', help='Directory of stop words.')
    parser.add_argument('--sample_number', type=int, default=5, help='Sample number for each bucket.')
    parser.add_argument('--threshold', type=float, default=0.3, help='Threshold for matching.')
    parser.add_argument('--name_len', type=int, default=9, help='Filename length.')
    parser.add_argument('--name_len_update', type=bool, default=False, help='To update file name length.')
    parser.add_argument('--lang', type=str, choices=['cn', 'en'], default='cn', help='Segmentor language setting.')
    args = parser.parse_args()
    return args


def main():
    args = _get_parser()

    # preliminary work
    check_file(args.infile)
    ensure_dir(args.output)

    if args.name_len_update:
        line_cnt = line_counter(args.infile)
        args.name_len = len(str(line_cnt)) + 1

    clean_dir(args.output, args.name_len)
    # end preliminary work

    p_bucket = defaultdict(list)
    save_idx = 0
    id_name = '{0:0' + str(args.name_len) + 'd}'
    # load stop words
    stop_words = get_stop_words(args.stop_words) if os.path.exists(args.stop_words) else list()
    # load tokenizer
    seg = Segmentor(args)

    print('Splitting sentence into different clusters ...')
    infile = open(args.infile, 'r', encoding="utf-8")
    for line in tqdm(infile):
        line = line.rstrip()
        is_match = False
        seg_list = list(seg.cut(line))
        if stop_words:
            seg_list = list(filter(lambda x: x not in stop_words, seg_list))
        for wd in seg_list:
            if is_match:
                break
            w_bucket = p_bucket[wd]
            for bucket in w_bucket:
                bucket_path = os.path.join(args.output, bucket)
                check_file(bucket_path)
                selected = sample_file(bucket_path, args.sample_number)
                selected = list(map(lambda x: list(seg.cut(x)), selected))
                # remove stop words
                if stop_words:
                    filt_selected = list()
                    for sen in selected:
                        sen = list(filter(lambda x: x not in stop_words, sen))
                        filt_selected.append(sen)
                    selected = filt_selected
                # calculate similarity with each bucket
                if all(jaccard(seg_list, cmp_list) > args.threshold for cmp_list in selected):
                    is_match = True
                    with open(bucket_path, 'a', encoding='utf-8') as outfile:
                        outfile.write(line+'\n')
                    for w in seg_list:
                        if bucket not in p_bucket[w]:
                            p_bucket[w].append(bucket)
                    break
        if not is_match:
            bucket_name = ('tmp' + id_name).format(save_idx)
            bucket_path = os.path.join(args.output, bucket_name)
            with open(bucket_path, 'a', encoding='utf-8') as outfile:
                outfile.write(line+'\n')
            for w in seg_list:
                p_bucket[w].append(bucket_name)
            save_idx += 1

    infile.close()

    # sort and rename file
    file_list = os.listdir(args.output)
    file_list = list(filter(lambda x: x.startswith('tmp'), file_list))
    cnt = dict()
    for file in file_list:
        file_path = os.path.join(args.output, file)
        cnt[file] = line_counter(file_path)

    sorted_cnt = sorted(cnt.items(), key=lambda kv: kv[1], reverse=True)
    for idx, (file_name, times) in enumerate(sorted_cnt):
        origin_path = os.path.join(args.output, file_name)
        new_path = os.path.join(args.output, id_name.format(idx))
        os.rename(origin_path, new_path)

    print('All is well')


if __name__ == '__main__':
    main()
