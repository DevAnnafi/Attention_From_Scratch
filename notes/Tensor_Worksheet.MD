# Day 1 — Tensor Mechanics Worksheet

Fill every blank from the PyTorch docs. Do not fill anything from memory or
guesswork — if you can't answer it, you haven't read the right page yet.

Rule for the whole day: **predict before you run.** Write the prediction, then
run, then reconcile. A wrong prediction is worth more than a right one if you
chase down why.

---

## Part 1 — The Reshape Family

Docs:
- https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html
- https://docs.pytorch.org/docs/stable/generated/torch.reshape.html
- https://docs.pytorch.org/docs/stable/generated/torch.transpose.html
- https://docs.pytorch.org/docs/stable/generated/torch.permute.html
- https://docs.pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html

### 1.1 One-sentence contracts

Write each in your own words. No copy-paste from the docs.

| Op | What it does | Returns a view or a copy? | Can it fail? When? |
|---|---|---|---|
| `view` |Reshapes the tensor without copying its data in memory |Always returns a view |It can fail when the view size is not the same as the original size or when the tensor's data is not stored in a contiguous layout. |
| `reshape` |Changes the shape of a tensor without changing its data |Returns a view |It fails when the requested shape does not match the total number of elements, or when you use the -1 trick incorrectly |
| `transpose` |Transpose changes the tensor by swapping dim1 and dim2 without changing its internal data |Returns a view |It can fail when you provide integers that are out of range or when you violate layout constraints on specific types like sparse or nested tensors |
| `permute` |Returns the tensor where the inner dimensions are changed |Returns a view |Fails when the dimensions are wrong, duplicated, out of range or have different types |
| `contiguous` |Groups all the tensor number together into a clean line in the computers memory |Returns a view |It can fail when the computer runs out of memory or if you use it on specialized tensor types like sparse layouts |

### 1.2 Questions to answer

1. Under exactly what condition does `view` raise an error?

   > When the elements arent the same and the stride

2. What does `reshape` do in that same situation that `view` won't?

   > View will throw a runtime error while reshape can alter tensor shapes even when the data is not contiguous in memory.

3. What can `permute` express that `transpose` cannot?

   > Permute can rearrange the number of dimensions simultaneously in a single step while transpose is strictly limited to swapping only two dimensions at a time.

4. What does "contiguous" mean in terms of how the data sits in memory?

   > The elements of a tensor sit in a single, unbroken, and sequential block of computer memory.

5. Why does `transpose` produce a non-contiguous tensor? (Think about what
   changed and what didn't.)

   > It is because the inner dimensions are being swapped 

6. What does calling `.contiguous()` actually cost?

   > The amount of memory being used

7. If `reshape` handles the non-contiguous case automatically, why would anyone
   write `.contiguous().view(...)` instead?

   > It is to intentionally force a runtime error if a tensor is non-contiguous preventing PyTorch from making silent, costly memory copies.

### 1.3 Prediction table

Start from `x = torch.arange(48).reshape(2, 3, 8)`. Fill the "Predicted" column
completely before running a single line.

| Expression | Predicted shape | Actual shape | Match? | If no — why? |
|---|---|---|---|---|
| `x.view(2, 24)` |[2,24]||[2,24] | |
| `x.view(6, 8)` |[6,8] |[6,8]| | |
| `x.transpose(0, 1)` |[3,2,8]|[3,2,8]| | |
| `x.transpose(1, 2)` |[2,8,3]|[2,8,3]| | |
| `x.permute(2, 0, 1)` |[8,2,3]|[8,2,3]| | |
| `x.view(2, 3, 2, 4)` |[2,3,2,4]|[2,3,2,4]| | |
| `x.view(2, 3, 2, 4).transpose(1, 2) |[2,2,3,4] |[2,2,3,4] | | |
| `x.unsqueeze(0)` |[2,3,8]|[1,2,3,8]| |I didn't realize it inserts an axis rather than modifying an existing one |
| `x.unsqueeze(2)` |[2,3,1,8]|[2,3,1,8] | | |

### 1.4 Trigger the error on purpose

Write code that makes `view` fail after a `transpose`. Paste the exact error
message here:

```
RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
```

Now fix it two different ways. Note what each one does differently:

- Fix A: x.transpose(0,2).reshape(3,4,2,2)
- Fix B: x.contiguous().view(3,4,2,2)

You will see this error again on Day 6. The point of writing it down is that you
recognize it in one second instead of ten minutes.

### 1.5 Round-trip test

Write a snippet that takes a `(2, 6, 8)` tensor, splits the last dim into
`(2, 6, 2, 4)`, transposes to `(2, 2, 6, 4)`, then reverses the whole thing back
to `(2, 6, 8)`.

Does `torch.allclose(original, round_tripped)` return True?

> True

If it doesn't, you have a bug — and it is the *exact* bug that breaks multi-head
attention. Fix it now while the test case is three lines long.

---

## Part 2 — Broadcasting

Doc: https://docs.pytorch.org/docs/stable/notes/broadcasting.html

### 2.1 State the rule

Write the broadcasting rule as a procedure you could run on paper, step by step:

1. When iterating over the dimension sizes, starting at the trailing dimension, the dimension sizes must be either be equal, one of them is 1, or one of them does not exist.
2. If the number of dimensions of x and y are not equal, add 1 to the dimensions of the tensor with fewer dimensions to make them equal length.
3. Then, for each dimension size, the resulting dimension size is the max of the sizes of x and y along that dimension.

### 2.2 Compatibility table

Fill "Predicted" first — including for the ones you think will fail.

| Shape A | Shape B | Compatible? | Predicted result shape | Actual |
|---|---|---|---|---|
| `(3, 4)` | `(4,)` |Yes| |(3,4)| (3,4)
| `(3, 4)` | `(3,)` | No | (3,4)| Runtime Error |
| `(2, 3, 4)` | `(3, 4)` | Yes | Error | (2,3,4)| | Was not an error because all dimensions 1 and 2 are the same and dimension 3 b has 1|
| `(2, 1, 4)` | `(3, 4)` | No | Error | (2,3,4)| |Was not an error because dimensions 1 are equal dimension 2 has a 1 in a|
| `(5, 1, 4, 1)` | `(3, 1, 7)` | Yes | (5,3,4,7)| (5,3,4,7)| |a has 1 b has 7 singleton; a has 4 b has 1 singleton; a has 1 b has 3 singleton; a has 5 b is empty does not exist clause |
| `(2, 3)` | `(3, 2)` | No | Error | Error |
| `(4, 1)` | `(1, 4)` | Yes| (4, 1)| (4,4)| |Trailing: a has 1, b has 4 — singleton clause, a stretches to 4. Leading: a has 4, b has 1 — singleton clause again, b stretches to 4. Both tensors expand, in different dimensions, and you land at (4, 4)| 

That last row is important. Look at the result shape and ask whether it's what a
person writing that line would have *wanted*.

### 2.3 The silent bug

Construct a case where two shapes broadcast successfully but the result is not
what you intended — no error raised, just wrong numbers.

```python
import torch

a = torch.tensor([[10], [20], [30]])  
b = torch.tensor([1, 2, 3])  

result = a + b

print(result)
```

What was the intent, and what actually happened?

> Wanted to perform a straightforward, element-by-element matching operation between two sets of three values. Instead of matching element-by-element, PyTorch performed a cross-product addition matrix, generating 9 elements instead of 3.

Why this matters: attention masks are the classic place this bites. A mask with
the wrong number of dimensions will broadcast happily and mask the wrong things.

---

## Part 3 — Batched Matmul

Doc: https://docs.pytorch.org/docs/stable/generated/torch.matmul.html

### 3.1 Questions

1. For inputs with more than 2 dimensions, which dims are treated as "batch" and
   which as the matrix?

   > The last two dimensions are the matrix and the dimensions before the last two are batch dimensions.

2. Do batch dimensions broadcast? What rule governs that?

   > Yes batch dimensions broadcast and the rule that governs that is that the first N-2 dimensions of each argument, the batch dimensions, are broadcast.

3. Write the general formula: given `(..., n, m)` matmul `(..., m, p)`, the
   output shape is:

   > given (j * l * n * m) matmul ( k * m * p), the outputs shape is (j * k * n * p)

### 3.2 Prediction table

| A shape | B shape | Predicted output | Actual | Notes |
|---|---|---|---|---|
| `(3, 4)` | `(4, 5)` | (3,5)| (3,5)| Matrix multiplication leads to the 2d tensor to become a 3*5 matrix|
| `(2, 3, 4)` | `(2, 4, 5)` |(2,3,5) | (2,3,5)| When the rows and columns match all you need to do is add the last column for the matrix and keep the first two of the first matrix|
| `(2, 3, 4)` | `(4, 5)` | (2,3,5) | (2,3,5) | Dot product still leads it to be a 2*3*5 matrix |
| `(2, 8, 6, 4)` | `(2, 8, 4, 6)` | (2,8,6,6)| (2,8,6,6)| Since height and weight are equal we just add the rest of the numbers as one ouput|
| `(2, 8, 6, 64)` | `(2, 8, 64, 6)` | (2,8,6,6) | (2,8,6,6)| | Height and weight are equal so we just add the rest of the numbers as one ouput
| `(2, 3, 4)` | `(2, 5, 6)` | Runtime Error| Runtime Error| First two dimensions don't match|

The second-to-last row is the shape of `QK^T` in multi-head attention. When you
get to Day 4, you should already know what comes out of it.

Paste the error from the last row here:

```
RuntimeError: Expected size for first two dimensions of batch2 tensor to be: [2, 4] but got: [2, 5].
```

---

## Part 4 — Softmax

Doc: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html

1. What does the `dim` argument select — precisely?

   > The dimension along which softmax will be computed

2. Given a tensor of attention scores shaped `(B, H, S_q, S_k)`, which `dim`
   makes each query's weights over all keys sum to 1?

   > dim=1

3. What would happen — mathematically, not as an error — if you used the wrong
   `dim` here?

   > It will sum to 1 along that axis but what is being compared and normalized is changed

Verify it: build a random `(2, 2, 5, 5)` tensor, softmax it along your chosen
dim, and write the assertion that proves each row sums to 1.

```python
import torch

a = torch.randn([2, 2, 5, 5])

softmax_result = torch.softmax(a, dim=1)

assert torch.allclose(softmax_result.sum(dim=1), torch.tensor(1.0))
```

---

## Part 5 — Reductions (for LayerNorm on Day 9)

1. What does `keepdim=True` change about the output shape?

   > The output tensor is of the same size as input but the dimensions change where it is of size 1.

2. Why would you want `keepdim=True` when computing a mean you're about to
   subtract from the original tensor?

   > Using keepdim=True keeps the number of dimensions the same when you compute a mean

Test: take a `(2, 5, 8)` tensor, compute its mean along the last dim with and
without `keepdim`, and record both shapes.

- Without: [2,5]
- With: [2,5,1]

---

## Part 6 — Closing Self-Quiz

Close every doc, close the scratch file, and answer on paper. No looking.

1. What condition makes `view` fail? When the view size is not compatible with input tensor's size and stride
2. What does `permute` give you over `transpose`? Permute allows for changing inner dimensions simultaneously while transpose only swaps dim1 and dim2.
3. Output shape of `(2, 8, 6, 64)` matmul `(2, 8, 64, 6)`? (2,8,6,6)
4. State the broadcasting rule from memory. The rule is for the dimensions it must either match, either one is a 1 or either one doesn't have to exist.
5. For attention scores `(B, H, S_q, S_k)`, which softmax dim and why? dim=-1 because each query's weights have to sum to 1 across all keys for the weighted aberage to be well-formed.

Anything shaky is your next 20 minutes. Go back to the doc page, not to the
answer above.

---

## Log

Three things that surprised you today. You'll want these on Day 6.

1. The softmax dim, which I got wrong but reasoned through it. It runs along the key axis so each query's weights sum to 1.
2. I thought since 1 does not equal to 3 it's not broadcastable.
3. The round-trip test where I thought I could just reshape it twice but I had to transpose twice to make it go back to its original shape and then apply view or reshape on it.
