from cs336_basics import BPETrainer, BPETokenizer

def main():
    trainer = BPETrainer(
        "/home/projects/CS336/assignment1-basics/data/chinese_text.txt", 
        500, 
        ["<|endoftext|>", "<|beginoftext|>"]
    )
    vocab, merges = trainer.train()

    # ========== 直观打印调试输出 ==========
    print("=" * 60)
    print(f"总词表大小: {len(vocab)}")
    print(f"总共生成合并规则数量: {len(merges)}")
    print("=" * 60)

    # 1. 打印前20条合并规则（按合并顺序）
    print("\n【前20条BPE合并规则】(byte1, byte2)")
    for idx, pair in enumerate(merges[:20]):
        b1, b2 = pair
        print(f"Merge {idx+1}: {b1} + {b2} → {b1+b2}")

    # 2. 打印词表末尾新增的合并子词（后50个，都是训练合并出来的token）
    print("\n【词表后100个新增子词token（合并生成）】ID -> bytes")
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[0])
    new_tokens = sorted_vocab[-100:]
    for token_id, byte_data in new_tokens:
        # 尝试转utf8字符串，无法解码则显示原始字节
        try:
            text = byte_data.decode("utf-8")
        except UnicodeDecodeError:
            text = f"raw_bytes:{list(byte_data)}"
        print(f"ID {token_id:4d} | bytes={byte_data!r} | text={text}")

    # 3. 用训练好的tokenizer编码测试文本，看分词效果（新增简易编码演示）
    tokenizer = BPETokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
    test_text = "<|beginoftext|>千里之行，始于足下,妮可妮可妮<|endoftext|>"
    token_ids = tokenizer.encode(test_text)
    print("\n【文本编码测试】")
    print(f"原始文本: {test_text}")
    print(f"编码token IDs: {token_ids}")

    # 反向解码看还原效果
    decode_text = tokenizer.decode(token_ids)
    print(f"解码还原文本: {decode_text}")

if __name__ == "__main__":
    main()