# Test file 2 for GPT implementation
# Tests for dataset creation and model components

import pytest
import tiktoken
import torch

from gpt_implementation import GPTDatasetV1, GPTModel


class TestGPTDatasetV2:
    """Test suite for GPTDatasetV1 class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.sample_text = "The quick brown fox jumps over the lazy dog. " * 10
        self.max_length = 4
        self.stride = 1
        
    def test_dataset_initialization(self):
        """Test dataset can be initialized properly"""
        dataset = GPTDatasetV1(
            txt=self.sample_text,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            stride=self.stride
        )
        assert len(dataset) > 0
        assert hasattr(dataset, 'input_ids')
        assert hasattr(dataset, 'target_ids')
        
    def test_dataset_getitem(self):
        """Test dataset returns correct format for items"""
        dataset = GPTDatasetV1(
            txt=self.sample_text,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            stride=self.stride
        )
        
        # Get first item
        item = dataset[0]
        
        # Check it returns a dictionary with correct keys
        assert isinstance(item, dict)
        assert 'input_ids' in item
        assert 'target_ids' in item
        
        # Check tensor shapes
        assert item['input_ids'].shape == torch.Size([self.max_length])
        assert item['target_ids'].shape == torch.Size([self.max_length])
        
    def test_dataset_length_calculation(self):
        """Test dataset length is calculated correctly"""
        short_text = "Hello world test"
        dataset = GPTDatasetV1(
            txt=short_text,
            tokenizer=self.tokenizer,
            max_length=4,
            stride=2
        )
        
        # Calculate expected length
        token_ids = self.tokenizer.encode(short_text)
        expected_length = max(0, (len(token_ids) - 4) // 2 + 1)
        
        assert len(dataset) == expected_length
        
    def test_input_target_relationship(self):
        """Test that target_ids are shifted input_ids"""
        dataset = GPTDatasetV1(
            txt="1 2 3 4 5 6 7 8",
            tokenizer=self.tokenizer,
            max_length=4,
            stride=1
        )
        
        if len(dataset) > 0:
            item = dataset[0]
            input_ids = item['input_ids']
            target_ids = item['target_ids']
            
            # Target should be input shifted by one position
            assert torch.equal(input_ids[1:], target_ids[:-1])


class TestGPTModelComponents:
    """Test suite for GPT model components"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = {
            "vocab_size": 50257,
            "context_length": 256,
            "emb_dim": 768,
            "n_heads": 12,
            "n_layers": 6,
            "drop_rate": 0.1,
            "qkv_bias": False
        }
        
    def test_model_initialization(self):
        """Test GPT model can be initialized"""
        model = GPTModel(self.config)
        
        # Check model has required attributes
        assert hasattr(model, 'tok_emb')
        assert hasattr(model, 'pos_emb')
        assert hasattr(model, 'trf_blocks')
        assert hasattr(model, 'final_norm')
        assert hasattr(model, 'out_head')
        
    def test_model_forward_pass_shape(self):
        """Test model forward pass produces correct output shape"""
        model = GPTModel(self.config)
        
        # Create dummy input
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, self.config["vocab_size"], (batch_size, seq_length))
        
        # Forward pass
        with torch.no_grad():
            output = model(input_ids)
            
        # Check output shape
        expected_shape = (batch_size, seq_length, self.config["vocab_size"])
        assert output.shape == expected_shape
        
    def test_model_with_different_sequence_lengths(self):
        """Test model handles different sequence lengths"""
        model = GPTModel(self.config)
        
        sequence_lengths = [1, 5, 16, 64]
        
        for seq_len in sequence_lengths:
            if seq_len <= self.config["context_length"]:
                input_ids = torch.randint(0, self.config["vocab_size"], (1, seq_len))
                
                with torch.no_grad():
                    output = model(input_ids)
                    
                assert output.shape == (1, seq_len, self.config["vocab_size"])
                
    def test_model_parameter_count(self):
        """Test model has reasonable parameter count"""
        model = GPTModel(self.config)
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Should have millions of parameters but not unreasonable
        assert total_params > 1_000_000  # At least 1M parameters
        assert total_params < 1_000_000_000  # Less than 1B parameters
        
    def test_model_training_mode(self):
        """Test model can switch between training and eval modes"""
        model = GPTModel(self.config)
        
        # Should start in training mode
        assert model.training
        
        # Switch to eval mode
        model.eval()
        assert not model.training
        
        # Switch back to training mode
        model.train()
        assert model.training


class TestTokenization:
    """Test suite for tokenization functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.tokenizer = tiktoken.get_encoding("gpt2")
        
    def test_basic_tokenization(self):
        """Test basic tokenization works"""
        text = "Hello, world!"
        tokens = self.tokenizer.encode(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(token, int) for token in tokens)
        
    def test_decode_encode_consistency(self):
        """Test that encoding then decoding returns original text"""
        text = "The quick brown fox jumps over the lazy dog."
        tokens = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(tokens)
        
        assert decoded == text
        
    def test_special_tokens(self):
        """Test handling of special tokens"""
        text = "Hello<|endoftext|>World"
        tokens = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
