"""
Infographic Generation Service for Executive Briefs.

Uses Replicate's image models to generate AI-powered infographics
summarizing portfolio/project metrics.
"""

import requests
from typing import Optional, Tuple


class InfographicService:
    """Generates AI-powered infographics using image models on Replicate."""

    # Available models on Replicate
    NANO_BANANA = "google/nano-banana-pro"  # Google's Nano Banana Pro
    FLUX_SCHNELL = "black-forest-labs/flux-schnell"  # Fast (~10-15 sec)
    FLUX_DEV = "black-forest-labs/flux-dev"  # Higher quality (~30-60 sec)

    def __init__(self, replicate_api_key: str, model: str = "nano-banana"):
        """
        Initialize the infographic service.

        Args:
            replicate_api_key: Replicate API key
            model: "nano-banana" (default), "schnell" (fast FLUX), or "dev" (high quality FLUX)
        """
        self.api_key = replicate_api_key.strip() if replicate_api_key else ""
        if not self.api_key:
            raise ValueError("Replicate API key is required")

        if model == "dev":
            self.model = self.FLUX_DEV
        elif model == "schnell":
            self.model = self.FLUX_SCHNELL
        else:
            self.model = self.NANO_BANANA  # Default to Nano Banana Pro

    def generate_portfolio_infographic(self, metrics: dict) -> Tuple[Optional[bytes], Optional[str]]:
        """
        Generate a portfolio infographic.

        Args:
            metrics: Dictionary containing portfolio metrics:
                - total_projects: int
                - cpi: float (Cost Performance Index)
                - spi: float (Schedule Performance Index)
                - critical_projects: int
                - healthy_pct: float (percentage)
                - at_risk_pct: float (percentage)
                - critical_pct: float (percentage)
                - total_budget: float (optional)
                - forecast_overrun: float (optional)

        Returns:
            Tuple of (image_bytes or None, error_message or None)
        """
        prompt = self._build_portfolio_prompt(metrics)
        return self._generate_image(prompt)

    def generate_project_infographic(self, metrics: dict) -> Tuple[Optional[bytes], Optional[str]]:
        """
        Generate a project infographic.

        Args:
            metrics: Dictionary containing project metrics:
                - project_name: str
                - cpi: float
                - spi: float
                - budget_pct: float (percentage of budget used)
                - time_pct: float (percentage of time elapsed)
                - health_status: str ("Healthy", "At Risk", "Critical")
                - completion_pct: float (optional)

        Returns:
            Tuple of (image_bytes or None, error_message or None)
        """
        prompt = self._build_project_prompt(metrics)
        return self._generate_image(prompt)

    def _build_portfolio_prompt(self, m: dict) -> str:
        """Build the prompt for portfolio infographic generation."""
        # Determine overall health status
        cpi = m.get('cpi', 1.0)
        spi = m.get('spi', 1.0)

        if cpi >= 1.0 and spi >= 1.0:
            health_status = "HEALTHY"
            health_icon = "green checkmark"
        elif cpi >= 0.9 or spi >= 0.9:
            health_status = "AT RISK"
            health_icon = "yellow warning"
        else:
            health_status = "CRITICAL"
            health_icon = "red alert"

        # CPI indicator
        cpi_indicator = "green checkmark" if cpi >= 1.0 else "yellow warning" if cpi >= 0.9 else "red alert"
        # SPI indicator
        spi_indicator = "green checkmark" if spi >= 1.0 else "yellow warning" if spi >= 0.9 else "red alert"

        # Health distribution
        healthy_pct = m.get('healthy_pct', 0)
        at_risk_pct = m.get('at_risk_pct', 0)
        critical_pct = m.get('critical_pct', 0)

        prompt = f"""Clean professional business infographic poster, white background, modern minimalist design, corporate style.

Title at top: "Portfolio Executive Summary"

Large metrics displayed with simple icons in a grid layout:
- "{m.get('total_projects', 0)} Projects" with building icon
- "CPI: {cpi:.2f}" with {cpi_indicator} icon
- "SPI: {spi:.2f}" with {spi_indicator} icon
- "{m.get('critical_projects', 0)} Critical" with red warning icon

Health status badge: {health_status} with {health_icon}

Simple donut chart or pie chart showing project distribution:
- {healthy_pct:.0f}% healthy (green)
- {at_risk_pct:.0f}% at risk (yellow/orange)
- {critical_pct:.0f}% critical (red)

Style: Corporate executive dashboard, data visualization, clean typography, professional presentation quality.
No decorative elements, simple flat icons, clear readable numbers.
"""
        return prompt

    def _build_project_prompt(self, m: dict) -> str:
        """Build the prompt for project infographic generation."""
        cpi = m.get('cpi', 1.0)
        spi = m.get('spi', 1.0)
        health_status = m.get('health_status', 'Unknown')

        # Determine color indicators
        cpi_color = "green" if cpi >= 1.0 else "yellow" if cpi >= 0.9 else "red"
        spi_color = "green" if spi >= 1.0 else "yellow" if spi >= 0.9 else "red"

        # Health status badge color
        if health_status == "Healthy":
            status_color = "green"
        elif health_status == "At Risk":
            status_color = "yellow"
        else:
            status_color = "red"

        project_name = m.get('project_name', 'Project')
        # Truncate long names for better display
        if len(project_name) > 30:
            project_name = project_name[:27] + "..."

        prompt = f"""Clean professional business infographic card, white background, modern flat design style.

Title at top: "{project_name}"

Key metrics with circular progress indicators arranged in 2x2 grid:
- "Budget Used: {m.get('budget_pct', 0):.0f}%" with circular progress bar
- "Schedule Used: {m.get('time_pct', 0):.0f}%" with circular progress bar
- "CPI: {cpi:.2f}" with {cpi_color} indicator dot
- "SPI: {spi:.2f}" with {spi_color} indicator dot

Status badge at bottom: "{health_status}" in {status_color} color

Completion: {m.get('completion_pct', 0):.0f}% shown as horizontal progress bar

Style: Corporate dashboard card, clean data visualization, executive presentation quality.
Simple flat icons, clear readable numbers, professional typography.
No decorative elements, minimalist design.
"""
        return prompt

    def _generate_image(self, prompt: str) -> Tuple[Optional[bytes], Optional[str]]:
        """
        Generate an image using image models on Replicate.

        Args:
            prompt: The image generation prompt

        Returns:
            Tuple of (image_bytes or None, error_message or None)
        """
        try:
            import replicate

            client = replicate.Client(api_token=self.api_key)

            # Build input parameters based on model
            if self.model == self.NANO_BANANA:
                # Nano Banana Pro parameters
                input_params = {
                    "prompt": prompt,
                    "aspect_ratio": "16:9",
                    "output_format": "png",
                }
            else:
                # FLUX model parameters
                input_params = {
                    "prompt": prompt,
                    "aspect_ratio": "16:9",
                    "output_format": "png",
                    "num_outputs": 1,
                }

            output = client.run(self.model, input=input_params)

            # Handle different output formats from Replicate
            image_url = None
            if isinstance(output, list) and len(output) > 0:
                # List of URLs or FileOutput objects
                first_output = output[0]
                if isinstance(first_output, str):
                    image_url = first_output
                elif hasattr(first_output, 'url'):
                    image_url = first_output.url
                else:
                    image_url = str(first_output)
            elif isinstance(output, str):
                image_url = output
            elif hasattr(output, 'url'):
                image_url = output.url
            elif hasattr(output, '__iter__'):
                # Iterator of FileOutput objects
                for item in output:
                    if hasattr(item, 'url'):
                        image_url = item.url
                        break
                    elif isinstance(item, str):
                        image_url = item
                        break

            if not image_url:
                return None, f"Unexpected output format from Replicate: {type(output)}"

            # Download the image
            resp = requests.get(image_url, timeout=120)
            resp.raise_for_status()
            return resp.content, None

        except ImportError:
            return None, "Replicate library not installed. Run: pip install replicate"
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"
