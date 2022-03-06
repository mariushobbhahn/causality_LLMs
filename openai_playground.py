#!/usr/bin/env python

import os
import openai

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_KEY")

response = openai.Completion.create(engine="text-davinci-001", prompt="The green ball hit the red ball. \
 The red ball hit the blue ball. The blue ball goes in the hole. Which ball was last?", max_tokens=10)
print(response)