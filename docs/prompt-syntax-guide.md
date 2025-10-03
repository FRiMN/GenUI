# Prompt Syntax Guide

This guide explains how to use advanced prompt features in GenUI, including Compel syntax for enhanced prompt control and the BREAK keyword for combining different prompt concepts.

## Overview

GenUI supports advanced prompt syntax that gives you more control over how your prompts are interpreted during image generation. These features allow you to:

- Emphasize or de-emphasize specific parts of your prompt
- Combine multiple concepts in a single prompt
- Create more precise and detailed image descriptions
- Have better control over the final output

## Basic Prompt Usage

Standard prompts work exactly as you expect:

```
a beautiful landscape with mountains and a lake
```

However, you can enhance your prompts using the advanced syntax described below.

## Compel Syntax Features

### 1. Weight Adjustment

You can adjust the importance (weight) of specific words or phrases in your prompt.

#### Increasing Emphasis

To make parts of your prompt more important, use parentheses with a weight value:

```
(beautiful woman)1.2 in a garden
```

- `1.0` = normal weight (default)
- `1.1` = slightly more emphasis
- `1.2` = moderate emphasis
- `1.5` = strong emphasis
- `2.0` = very strong emphasis

#### Alternative Syntax for Emphasis

You can also use `+` symbols for quick emphasis:

```
(beautiful woman)+ in a garden     # equivalent to 1.1
(beautiful woman)++ in a garden    # equivalent to 1.21
(beautiful woman)+++ in a garden   # equivalent to 1.33
```

#### Reducing Emphasis

To make parts less important, use weights below 1.0 or `-` symbols:

```
portrait of a woman, (busy background)0.8
```

Using minus symbols:
```
portrait of a woman, (busy background)-
portrait of a woman, (busy background)--    # even less emphasis
```

### 2. Complex Weight Combinations

You can combine multiple weighted elements:

```
(beautiful portrait)1.3 of a (young woman)1.1 with (blue eyes)1.2, (detailed background)0.8
```

## BREAK Keyword - Combining Multiple Concepts

The `BREAK` keyword allows you to combine different concepts or scenes in your prompt while maintaining control over each part.

### Basic BREAK Usage

Use `BREAK` to separate different concepts in your prompt:

```
portrait of a young woman BREAK sitting in a beautiful garden BREAK impressionist painting style
```

This tells the AI to consider three distinct parts:
1. The subject: "portrait of a young woman"
2. The setting: "sitting in a beautiful garden"
3. The style: "impressionist painting style"

### When to Use BREAK

BREAK is particularly useful for:

**Complex Scenes:**
```
medieval castle on a hill BREAK dramatic storm clouds BREAK golden hour lighting
```

**Character + Environment:**
```
warrior in armor holding a sword BREAK ancient battlefield with ruins BREAK epic fantasy art style
```

**Multiple Subjects:**
```
two people having a conversation BREAK cozy coffee shop interior BREAK warm afternoon lighting
```

**Style Separation:**
```
portrait of an elegant woman BREAK ornate Victorian dress BREAK painted in oil painting style
```

### BREAK with Commas

You can use commas after BREAK for readability:

```
portrait of a woman BREAK, standing in a field of flowers BREAK, sunset lighting
```

Both formats work the same way - use whichever feels more natural to you.

### Combining BREAK with Weights

You can use Compel weighting with BREAK sections:

```
(beautiful portrait)1.2 of a woman BREAK (magical forest)1.1 background BREAK (fantasy art style)0.9
```

## Practical Examples

### Example 1: Portrait with Environment
```
(detailed portrait)1.2 of a young woman with long hair BREAK sitting in a (cozy library)1.1 BREAK (warm lighting)1.3, books in background
```

### Example 2: Fantasy Scene
```
mighty dragon perched on a cliff BREAK (medieval castle)1.1 in the distance BREAK (dramatic sunset)1.2 with orange and purple clouds
```

### Example 3: Artistic Style Control
```
portrait of an elderly man BREAK (intricate details)1.2 on weathered face BREAK painted in (oil painting style)1.3, (soft brush strokes)1.1
```

### Example 4: Complex Scene Composition
```
(two characters)1.2 talking in a marketplace BREAK bustling crowd in background BREAK (medieval fantasy setting)1.1 BREAK (warm daylight)1.2
```

## Best Practices

### 1. Start Simple
Begin with basic prompts and gradually add complexity:
- Start: `woman in a garden`
- Add weights: `(beautiful woman)1.1 in a (lush garden)1.2`
- Add BREAK: `(beautiful woman)1.1 BREAK (lush garden)1.2 BREAK soft lighting`

### 2. Don't Overuse Weights
- Use weights sparingly - too many can conflict with each other
- Most weights should be between 0.8 and 1.3
- Extreme weights (above 1.5 or below 0.5) can produce unexpected results

### 3. Logical BREAK Sections
Group related concepts together in each BREAK section:
```
# Good
main character BREAK environment description BREAK artistic style

# Less effective
main character BREAK single adjective BREAK another adjective
```

### 4. Balance Your Sections
Try to make each BREAK section contribute meaningfully to the image:
```
# Balanced
detailed portrait BREAK ornate background BREAK dramatic lighting

# Unbalanced
(extremely detailed portrait)2.0 BREAK simple background BREAK basic lighting
```

### 5. Test and Iterate
- Generate with simple prompts first to establish a baseline
- Add one enhancement at a time to see its effect
- Keep notes on which techniques work best for your style

## Common Patterns

### Portrait Photography Style
```
(professional headshot)1.2 of a person BREAK (studio lighting)1.1 BREAK (shallow depth of field)1.2
```

### Landscape Photography Style
```
(majestic mountain landscape)1.2 BREAK (golden hour lighting)1.3 BREAK (wide angle view)1.1
```

### Fantasy Art Style
```
(epic fantasy character)1.2 BREAK (magical environment)1.1 BREAK (digital art style)1.2, highly detailed
```

### Artistic Painting Style
```
(expressive portrait)1.1 BREAK (textured brushstrokes)1.2 BREAK painted in (impressionist style)1.3
```

## Troubleshooting

### If Your Results Are Too Intense
- Reduce weight values (try 1.1 instead of 1.3)
- Use fewer weighted terms
- Check that weights aren't conflicting

### If BREAK Isn't Working as Expected
- Make sure each section has substantial content
- Try spacing out your BREAK sections more
- Ensure punctuation is correct

### If Prompts Are Being Cut Off
- GenUI no longer truncates long prompts automatically
- Very long prompts are fully supported
- Break up extremely long prompts with BREAK for better organization

## Advanced Tips

### Negative Prompts
All these features work in negative prompts too:
```
Negative: (blurry)1.3, (low quality)1.2 BREAK (bad anatomy)1.1 BREAK (oversaturated colors)1.2
```

### Experimentation
- Try different weight values to find what works for your style
- Experiment with different BREAK arrangements
- Save successful prompt patterns for future use

### Combining Techniques
You can combine all features for maximum control:
```
(masterpiece)1.2, (detailed portrait)1.3 of a (elegant woman)1.1 BREAK sitting in a (ornate chair)1.1 in a (luxurious room)1.2 BREAK (soft lighting)1.2, (oil painting style)1.1
```

Remember, these are tools to help you achieve your creative vision. Start simple and build complexity as you become more comfortable with the syntax.
