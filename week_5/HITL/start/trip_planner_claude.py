"""
Trip Planning System with Human-in-the-Loop

This script demonstrates a simple HITL system for planning a trip with three steps:
1. Destination selection with LLM suggestion
2. Flight selection with LLM suggestion
3. Itinerary generation with LLM suggestion

Each step includes human verification and modification options.
"""

import json
from typing import Dict, Optional, List
from dataclasses import dataclass
import os

from langsmith.utils import get_host_url
from pydantic import BaseModel, Field
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

@dataclass
class TripPlan:
    """Represents a complete trip plan."""
    destination: str
    departure_city: str
    flight_info: Dict
    itinerary: Dict

class FlightInfo(BaseModel):
    """Model for flight information."""
    flight_number: str = Field(description="Flight number")
    departure_time: str = Field(description="Departure time in HH:MM format")
    arrival_time: str = Field(description="Arrival time in HH:MM format")
    airline: str = Field(description="Name of the airline")

class Itinerary(BaseModel):
    """Model for daily itinerary."""
    daily_activities: List[List[str]] = Field(description="List of activities for each day of the trip")

class TripPlanner:
    def __init__(self):
        self.client = Anthropic()
        self.trip_plan = None

    def suggest_destination(self) -> str:
        """Use LLM to suggest a travel destination."""
        prompt = """Suggest an interesting travel destination. 
        Consider factors like weather, tourist attractions, and cultural experiences.
        Return only the destination name, nothing else."""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-5",
                max_tokens=1000,
                system="You are a helpful travel assistant.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"Error getting destination suggestion: {e}")
            return "Paris"  # Fallback destination

    def suggest_flight(self, departure_city: str, destination: str, user_prompt: str = None) -> Dict:
        """Use LLM to suggest a fictional flight."""
        prompt = f"""Generate a fictional flight from {departure_city} to {destination}.
        Include departure time, arrival time, and flight number."""

        messages = []
        messages.append({"role": "user", "content": prompt})
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})


        try:
            response = self.client.messages.parse(
                model="claude-haiku-4-5",
                max_tokens=1000,
                system="You are a helpful travel assistant.",
                messages=messages,
                output_format=FlightInfo
            )
            return response.parsed_output
        except Exception as e:
            print(f"Error getting flight suggestion: {e}")
            return {
                "flight_number": "AA123",
                "departure_time": "10:00",
                "arrival_time": "12:00",
                "airline": "Example Airlines"
            }

    def generate_itinerary(self, destination: str, user_prompt: str = None) -> Dict:
        """Use LLM to generate a brief itinerary."""
        prompt = f"""Generate a brief 3-day itinerary for {destination}.
        Include 2-3 activities per day."""

        messages = []
        messages.append({"role": "user", "content": prompt})
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})

        try:
            response =self.client.messages.parse(
                model="claude-haiku-4-5",
                max_tokens=1000,
                system="You are a helpful travel assistant.",
                messages= messages,
                output_format=Itinerary
            )
            return response.parsed_output
        except Exception as e:
            print(f"Error generating itinerary: {e}")
            return {
                "day1": ["Morning activity", "Afternoon activity"],
                "day2": ["Morning activity", "Afternoon activity"],
                "day3": ["Morning activity", "Afternoon activity"]
            }

    def get_human_confirmation(self, prompt: str, input_need: bool = False) -> bool:
        """Get human confirmation for a suggestion."""
        while True:
            choice = input(f"{prompt} (y/n): ").lower().strip()
            if choice in ['y', 'n']:
                return choice == 'y'
            print("Please enter 'y' or 'n'")

    def get_human_feedback(self, prompt: str) -> str:
        """Get human feedback for a suggestion."""
        return input(f"{prompt} : ").lower().strip()


    def plan_trip(self) -> None:
        """Run the complete trip planning process."""
        print("\n=== Welcome to the Trip Planner! ===\n")

        ## TODO Decide which steps would benefit from a human-in-the-loop experience.
        # Use get_human_confirmation to stop the flow and ask the human a question.

        # Step 1: Destination Selection
        suggested_destination = self.suggest_destination()
        print(f"\nSuggested destination: {suggested_destination}")
        
        # Example
        if self.get_human_confirmation("Would you like to use this destination?"):
            destination = suggested_destination
        else:
            destination = input("Enter your preferred destination: ").strip()
        
        # Step 2: Flight Selection
        departure_city = input("\nEnter your departure city: ").strip()
        suggested_flight = self.suggest_flight(departure_city, destination)
        
        print("\nSuggested flight:")
        print(f"Airline: {suggested_flight.airline}")
        print(f"Flight: {suggested_flight.flight_number}")
        print(f"Departure: {suggested_flight.departure_time}")
        print(f"Arrival: {suggested_flight.arrival_time}")

        while True:
            if self.get_human_confirmation("Will this flight schedule works for you?"):
                break
            else:

                prompt = self.get_human_feedback("do you have any preferred time or schedule?")
                suggested_flight = self.suggest_flight(departure_city, destination, prompt)

                print("\nSuggested flight:")
                print(f"Airline: {suggested_flight.airline}")
                print(f"Flight: {suggested_flight.flight_number}")
                print(f"Departure: {suggested_flight.departure_time}")
                print(f"Arrival: {suggested_flight.arrival_time}")
        
        # Step 3: Itinerary Generation
        suggested_itinerary = self.generate_itinerary(destination)
        
        print("\nSuggested itinerary:")
        for index, activities in enumerate(suggested_itinerary.daily_activities, 1):
            print(f"\nDAY {index + 1}:")
            for activity in activities:
                print(f"- {activity}")

        while True:
            if self.get_human_confirmation("Will this itinerary schedule works for you?"):
                break
            else:
                prompt = self.get_human_feedback("do you have any preferred time or change the schedule?")
                suggested_itinerary = self.generate_itinerary(destination, prompt)

                print("\nSuggested itinerary:")
                for index, activities in enumerate(suggested_itinerary.daily_activities, 1):
                    print(f"\nDAY {index + 1}:")
                    for activity in activities:
                        print(f"- {activity}")

        # Save the complete trip plan
        self.trip_plan = TripPlan(
            destination=destination,
            departure_city=departure_city,
            flight_info=suggested_flight,
            itinerary=suggested_itinerary
        )

        # Display final plan
        print("\n=== Your Trip Plan ===")
        print(f"Destination: {self.trip_plan.destination}")
        print(f"Departure from: {self.trip_plan.departure_city}")
        print("\nFlight Information:")
        print(f"Airline: {suggested_flight.airline}")
        print(f"Flight: {suggested_flight.flight_number}")
        print(f"Departure: {suggested_flight.departure_time}")
        print(f"Arrival: {suggested_flight.arrival_time}")
        print("\nItinerary:")
        for index, activities in enumerate(self.trip_plan.itinerary.daily_activities, 1):
            print(f"\nDAY {index + 1}:")
            for activity in activities:
                print(f"- {activity}")

def main():
    planner = TripPlanner()
    planner.plan_trip()

if __name__ == "__main__":
    main() 