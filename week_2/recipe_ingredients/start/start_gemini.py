
from pydantic import BaseModel, Field
from typing import List
from pprint import pprint
import os
from google import genai
import keys

class Ingredient(BaseModel):
    """
    Use this model to list ingredients for the recipe
    """
    name: str = Field(description="Name of the ingredient")
    unit: str = Field(description="Unit of the ingredient ex: tablespoon, cup, grams, ounce")
    amount: float = Field(description="Amount of the ingredient")


class Recipe(BaseModel):
    """
    Use this model when working with complete cooking recipes.
    """
    title: str = Field(description="Name of the recipe")
    ingredients: List[Ingredient] = Field(description="List of ingredients needed for the recipe")
    instructions: List[str] = Field(description="Step-by-step instructions to prepare the recipe")


def get_recipe_from_text(recipe_text: str) -> Recipe:
    """
    Convert recipe text into a structured Recipe object using OpenAI.
    """
    client = genai.Client(api_key=keys.api_key)

    # Make the API call
    response = client.interactions.create(
        model="gemini-3.1-flash-lite",
        input=f"""Convert this recipe into the specified format:\n\n{recipe_text}
                Also provide the units in 'grams, cup, ounce or teaspoon'
              """,
        response_format= {
            "type": "text",
            "mime_type": "application/json",
            "schema": Recipe.model_json_schema()
        }
    )

    res = Recipe.model_validate_json(json_data=response.output_text)
    print(res)
    return res


# Example usage
if __name__ == "__main__":
    # Read recipe text from file

    script_dir = os.path.dirname(os.path.abspath(__file__))
    recipe_path = os.path.join(script_dir, "mac_and_cheese_recipe.txt")
    with open(recipe_path, "r") as file:
        recipe_text = file.read()

    # Get structured recipe
    recipe = get_recipe_from_text(recipe_text)

    # Print results
    pprint(recipe.ingredients[0])
    # pprint(recipe) # to see the whole object