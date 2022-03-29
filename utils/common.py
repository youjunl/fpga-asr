import torch
from typing import List


def __ctc_decoder_predictions_tensor(tensor, labels):
    """
    Decodes a sequence of labels to words
    """
    blank_id = len(labels)
    hypotheses = []
    labels_map = dict([(i, labels[i]) for i in range(len(labels))])
    prediction_cpu_tensor = tensor.long().cpu()
    # iterate over batch
    for ind in range(prediction_cpu_tensor.shape[0]):
        # print(prediction)
        prediction = prediction_cpu_tensor[ind].numpy().tolist()
        # CTC decoding procedure
        decoded_prediction = []
        previous = len(labels)  # id of a blank symbol
        for p in prediction:
            if (p != previous or previous == blank_id) and p != blank_id:
                decoded_prediction.append(p)
            previous = p
        hypothesis = ''.join([labels_map[c] for c in decoded_prediction])
        hypotheses.append(hypothesis)
    return hypotheses




def __gather_losses(losses_list: list) -> list:
    return [torch.mean(torch.stack(losses_list))]


def __gather_predictions(predictions_list: list, labels: list) -> list:
    results = []
    for prediction in predictions_list:
        results += __ctc_decoder_predictions_tensor(prediction, labels=labels)
    return results


def __gather_transcripts(transcript_list: list, transcript_len_list: list,
                         labels: list) -> list:
    results = []
    labels_map = dict([(i, labels[i]) for i in range(len(labels))])
    # iterate over workers
    for t, ln in zip(transcript_list, transcript_len_list):
        # iterate over batch
        t_lc = t.long().cpu()
        ln_lc = ln.long().cpu()
        for ind in range(t.shape[0]):
            tgt_len = ln_lc[ind].item()
            target = t_lc[ind][:tgt_len].numpy().tolist()
            reference = ''.join([labels_map[c] for c in target])
            results.append(reference)
    return results


def process_evaluation_batch(tensors: dict, global_vars: dict, labels: list):
    """
    Creates a dictionary holding the results from a batch of audio
    """
    if 'EvalLoss' not in global_vars.keys():
        global_vars['EvalLoss'] = []
    if 'predictions' not in global_vars.keys():
        global_vars['predictions'] = []
    if 'transcripts' not in global_vars.keys():
        global_vars['transcripts'] = []
    if 'logits' not in global_vars.keys():
        global_vars['logits'] = []
    # if not 'transcript_lengths' in global_vars.keys():
    #  global_vars['transcript_lengths'] = []
    for kv, v in tensors.items():
        if kv.startswith('loss'):
            global_vars['EvalLoss'] += __gather_losses(v)
        elif kv.startswith('predictions'):
            global_vars['predictions'] += __gather_predictions(
                v, labels=labels)
        elif kv.startswith('transcript_length'):
            transcript_len_list = v
        elif kv.startswith('transcript'):
            transcript_list = v
        elif kv.startswith('output'):
            global_vars['logits'] += v

    global_vars['transcripts'] += __gather_transcripts(transcript_list,
                                                       transcript_len_list,
                                                       labels=labels)


def process_evaluation_epoch(global_vars: dict,
                             eval_metric='WER',
                             tag=None,
                             logger=None):
    """
    Calculates the aggregated loss and WER across the entire evaluation dataset
    """
    eloss = torch.mean(torch.stack(global_vars['EvalLoss'])).item()
    hypotheses = global_vars['predictions']
    references = global_vars['transcripts']

    eval_metric = eval_metric.upper()
    if eval_metric not in {'WER', 'CER'}:
        raise ValueError('eval_metric must be \'WER\' or \'CER\'')
    use_cer = True if eval_metric == 'CER' else False

    wer = word_error_rate(hypotheses=hypotheses,
                          references=references,
                          use_cer=use_cer)

    if tag is None:
        if logger:
            logger.info(f"==========>>>>>>Evaluation Loss: {eloss}")
            logger.info(f"==========>>>>>>Evaluation {eval_metric}: "
                        f"{wer*100 : 5.2f}%")
        else:
            print(f"==========>>>>>>Evaluation Loss: {eloss}")
            print(f"==========>>>>>>Evaluation {eval_metric}: "
                  f"{wer*100 : 5.2f}%")
        return {"Evaluation_Loss": eloss, f"Evaluation_{eval_metric}": wer}
    else:
        if logger:
            logger.info(f"==========>>>>>>Evaluation Loss {tag}: {eloss}")
            logger.info(f"==========>>>>>>Evaluation {eval_metric} {tag}: "
                        f"{wer*100 : 5.2f}%")
        else:
            print(f"==========>>>>>>Evaluation Loss {tag}: {eloss}")
            print(f"==========>>>>>>Evaluation {eval_metric} {tag}:"
                  f" {wer*100 : 5.2f}%")
        return {f"Evaluation_Loss_{tag}": eloss,
                f"Evaluation_{eval_metric}_{tag}": wer}


def post_process_predictions(predictions, labels):
    return __gather_predictions(predictions, labels=labels)


def post_process_transcripts(
        transcript_list, transcript_len_list, labels):
    return __gather_transcripts(transcript_list,
                                transcript_len_list,
                                labels=labels)

def __levenshtein(a: List, b: List) -> int:
    """Calculates the Levenshtein distance between a and b.
    The code was copied from: http://hetland.org/coding/python/levenshtein.py
    """
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


def word_error_rate(hypotheses: List[str],
                    references: List[str],
                    use_cer=False) -> float:
    """
    Computes Average Word Error rate between two texts represented as
    corresponding lists of string. Hypotheses and references must have same
    length.

    Args:
      hypotheses: list of hypotheses
      references: list of references
      use_cer: bool, set True to enable cer
    Returns:
      (float) average word error rate
    """
    scores = 0
    words = 0
    if len(hypotheses) != len(references):
        raise ValueError(
            "In word error rate calculation, hypotheses and reference"
            " lists must have the same number of elements. But I got:"
            "{0} and {1} correspondingly".format(len(hypotheses),
                                                 len(references)))
    for h, r in zip(hypotheses, references):
        if use_cer:
            h_list = list(h)
            r_list = list(r)
        else:
            h_list = h.split()
            r_list = r.split()
        words += len(r_list)
        scores += __levenshtein(h_list, r_list)
    if words != 0:
        wer = 1.0 * scores / words
    else:
        wer = float('inf')
    return wer

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# def setup_transcribe_dataloader(cfg):
#     config = {
#         'manifest_filepath': cfg['manifest_filepath'],
#         'sample_rate': cfg['sample_rate'],
#         'labels': cfg['labels'],
#         'batch_size': cfg['batch_size'],
#         'trim_silence': True,
#         'shuffle': False,
#     }
#     dataset = AudioToCharDataset(
#         manifest_filepath=config['manifest_filepath'],
#         labels=config['labels'],
#         sample_rate=config['sample_rate'],
#         int_values=config.get('int_values', False),
#         augmentor=None,
#         max_duration=config.get('max_duration', None),
#         min_duration=config.get('min_duration', None),
#         max_utts=config.get('max_utts', 0),
#         blank_index=config.get('blank_index', -1),
#         unk_index=config.get('unk_index', -1),
#         normalize=config.get('normalize_transcripts', False),
#         trim=config.get('trim_silence', True),
#         parser=config.get('parser', 'en'),
#     )
#     return torch.utils.data.DataLoader(
#         dataset=dataset,
#         batch_size=config['batch_size'],
#         collate_fn=dataset.collate_fn,
#         drop_last=config.get('drop_last', False),
#         shuffle=False,
#         num_workers=config.get('num_workers', 0),
#         pin_memory=config.get('pin_memory', False),
#     )

def ctc_decoder(tensor, labels):
    """
    Decodes a sequence of labels to words
    """
    blank_id = len(labels)
    labels_map = dict([(i, labels[i]) for i in range(len(labels))])
    prediction_cpu_tensor = tensor.long().cpu()
    # iterate over batch

    # print(prediction)
    prediction = prediction_cpu_tensor.numpy().tolist()
    # CTC decoding procedure
    decoded_prediction = []
    previous = len(labels)  # id of a blank symbol
    for p in prediction:
        if (p != previous or previous == blank_id) and p != blank_id:
            decoded_prediction.append(p)
        previous = p
    hypothesis = ''.join([labels_map[c] for c in decoded_prediction])

    return hypothesis