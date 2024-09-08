import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
def compute_advantages(rewards, values, gamma=0.99, normalize=True):
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    # Compute returns (discounted sum of rewards)
    returns[-1] = rewards[-1]
    for t in reversed(range(T - 1)):
        returns[t] = rewards[t] + gamma * returns[t + 1]

    # Compute advantages
    advantages = returns - values

    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages

def ppo_loss(old_probs, new_probs, advantages, epsilon=0.2):
    old_probs = torch.sum(old_probs,dim=(1,2))
    new_probs = torch.sum(new_probs,dim=(1,2))
    ratio = new_probs / old_probs
    advantages = advantages.to(ratio.device)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    surrogate1 = ratio * advantages
    surrogate2 = clipped_ratio * advantages
    loss = -torch.min(surrogate1, surrogate2)
    return loss.mean()

def token_wise_reward(generated_tokens_tensor, reference_tokens_tensor):
    smooth = 1e-8  # Smoothing value to prevent division by zero
    tokenwise_bleu_scores = []

    batch_size, max_seq_length = generated_tokens_tensor.size()

    for i in range(batch_size):
        gen_tokens = generated_tokens_tensor[i]  # Get tokens for the i-th sequence in the batch
        ref_tokens = reference_tokens_tensor[i]  # Get tokens for the i-th sequence in the batch

        gen_len = torch.sum(gen_tokens != 0).item()  # Length of the generated sequence
        ref_len = torch.sum(ref_tokens != 0).item()  # Length of the reference sequence

        # Calculate precision for each token
        precisions = []
        for gen_token in gen_tokens[:gen_len]:  # Iterate over tokens in the generated sequence
            token = gen_token.item()
            if token in ref_tokens:
                gen_count = torch.sum(gen_tokens == token).item()
                ref_count = torch.sum(ref_tokens == token).item()
                precision = min(gen_count, ref_count) / max(gen_count, ref_count)
                precisions.append(precision)
            else:
                precisions.append(0.0)

        # Calculate BLEU score for the sequence
        bleu_score = torch.exp(torch.tensor(1.0 / gen_len) * torch.sum(torch.log(torch.tensor(precisions)) + smooth))
        tokenwise_bleu_scores.append(bleu_score)

    return torch.stack(tokenwise_bleu_scores)


