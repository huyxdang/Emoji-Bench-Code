Because we want to simulate the real world – whereby a model keeps on working on a task without being prompted to check for errors. 
Therefore, what we can do is. Let X be the total number of steps, and Y < X, and within steps Y contains an error. Then, we have: 
First step: 
For a question of X steps, we give LLM the first Y steps (assistant) 
Then, for 2nd user step: “Please continue working” 
2nd assistant step: “
We evaluate whether the LLM can detect the error, and would still be able to correct the error and finally get the correct answer? Or does it simply continue from the error and continue hallucinate? 
Basically, we’re detecting whether the LLM can “self-detect” that it’s made an error, without being prompted. 

More specifically,

Procedure
Turn 1 (User): "Here is a formal system with the following operation table: [table]. Simplify the following expression: [expression]"
Turn 1 (Assistant — prefilled): Steps 1 through Y, where step Y contains an incorrect table lookup. Written in natural chain-of-thought style ("Let me work through this step by step...")
Turn 2 (User): "Please continue."
Turn 2 (Assistant — model generates): Model continues from where the prefill left off. This is what we evaluate.
Metrics
Self-detection Rate (%): Percentage model explicitly flags an error in the pre-filled steps (unprompted) 
Correction rate (%): Of those that detect, % that correctly fix the error and reach the correct final answer
Evaluation
For metric (1), use regex for {error, wait, mistake, correction, etc.}. → Use LLM-as-a-judge only for ambiguous cases. 
For metric (2), we can prompt the LLM since the very beginning to output the final output as “Say your final output as Final Output: X.” Then, use regex to extract the answer.
Note
We must ensure that the case is NOT convergent. I.e., ground_truth != final_output_w_error
Sample = ~100 samples only. 
