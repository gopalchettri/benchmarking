async def evaluate_criteria(
    self,
    llm_content: List[Dict[str, Any]],
    test_case: Dict[str, Any],
    model_name: str
) -> EvaluationResult:
    """
    Main evaluation method - calculates all metrics asynchronously.
    
    Args:
        llm_content: List[Dict] with format [{"id": int, "criteria": str}, ...]
        test_case: Dict containing compare_content with expected_content
        model_name: Name of the model being evaluated (for records)
        
    Returns:
        EvaluationResult with all computed metrics
    """
    
    # Validate inputs
    if not isinstance(llm_content, list):
        raise ValueError("llm_content must be a list")
    if not isinstance(test_case, dict):
        raise ValueError("test_case must be a dictionary")
    
    compare_content = test_case.get("compare_content", {})
    expected_content = compare_content.get("expected_content", [])
    
    if not llm_content or not expected_content:
        logger.warning("Empty llm_content or expected_content, returning empty result")
        return self._create_empty_result(test_case, model_name)
    
    # Extract criteria strings from both (identical format)
    llm_criteria_list = [item["criteria"] for item in llm_content if "criteria" in item]
    expected_criteria_list = [item["criteria"] for item in expected_content if "criteria" in item]
    
    if not llm_criteria_list or not expected_criteria_list:
        logger.warning("No valid criteria found after extraction, returning empty result")
        return self._create_empty_result(test_case, model_name)
    
    # Pre-compute embeddings once (awaits model load if needed)
    llm_embeds, exp_embeds = await self._compute_embeddings_for_both(
        llm_criteria_list, expected_criteria_list
    )
    
    # Run concurrent metric computations
    try:
        scores = await asyncio.gather(
            self._calculate_semantic_similarity(llm_embeds, exp_embeds),
            self._calculate_exact_match(llm_criteria_list, expected_criteria_list),
            self._calculate_completeness(llm_embeds, exp_embeds, compare_content),
            self._calculate_precision(llm_embeds, exp_embeds),
            self._calculate_keyword_coverage(llm_criteria_list, expected_criteria_list),
            return_exceptions=True
        )
        
        # Handle exceptions in concurrent results
        for i, score in enumerate(scores):
            if isinstance(score, Exception):
                logger.error(f"Error calculating metric {i}: {score}", exc_info=True)
                scores[i] = 0.0
        
        semantic_sim, exact_match, completeness, precision, keyword_cov = scores
        
    except Exception as e:
        logger.error(f"Error during metric computation: {e}", exc_info=True)
        return self._create_empty_result(test_case, model_name)
    
    # Calculate composite metrics
    f1_score = self._calculate_f1(precision, completeness)
    composite_score = self._calculate_composite(
        semantic_sim, exact_match, completeness, precision, keyword_cov
    )
    
    return EvaluationResult(
        control_id=test_case.get("control_id", "unknown"),
        model=model_name,
        framework=test_case.get("framework", ""),
        semantic_similarity=round(semantic_sim, 4),
        exact_match_score=round(exact_match, 4),
        completeness_recall=round(completeness, 4),
        precision=round(precision, 4),
        keyword_coverage=round(keyword_cov, 4),
        f1_score=round(f1_score, 4),
        composite_score=round(composite_score, 4),
        expected_count=compare_content.get("expected_count", len(expected_criteria_list)),
        generated_count=len(llm_criteria_list)
    )


async def _compute_embeddings_for_both(
    self,
    llm_criteria: List[str],
    expected_criteria: List[str]
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Compute embeddings for both LLM-generated and expected content in one batch.
    
    Args:
        llm_criteria: List of criteria strings from LLM
        expected_criteria: List of expected criteria strings
        
    Returns:
        Tuple of (llm_embeddings, expected_embeddings)
    """
    # Combine both lists for batch processing (more efficient)
    all_texts = llm_criteria + expected_criteria
    
    # Compute embeddings for all texts
    all_embeddings = await self.compute_embeddings_batch(all_texts)
    
    # Split embeddings back into two lists
    llm_embeddings = all_embeddings[:len(llm_criteria)]
    expected_embeddings = all_embeddings[len(llm_criteria):]
    
    return llm_embeddings, expected_embeddings


async def compute_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
    """
    Compute embeddings for a batch of text strings.
    Uses caching for performance optimization.
    
    Args:
        texts: List of strings to embed
        
    Returns:
        List of embedding vectors
    """
    # Create cache key
    cache_key = str(hash("||".join(texts)))
    
    # Check cache
    if cache_key in self._embedding_cache:
        logger.debug(f"Cache hit for {len(texts)} texts")
        return self._embedding_cache[cache_key]
    
    # Compute embeddings
    embeddings = self.embedding_model.encode(
        texts,
        convert_to_tensor=False,
        show_progress_bar=False,
        batch_size=32
    ).tolist()
    
    # Cache results
    self._embedding_cache[cache_key] = embeddings
    logger.debug(f"Computed and cached embeddings for {len(texts)} texts")
    
    return embeddings
