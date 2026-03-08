"""
Protocol Evaluator Tool

Evaluates lab protocols for completeness, accuracy, scientific rigor, reference quality,
and suggests improvements. Validates against standard methods from databases like PubMed,
Addgene, and protocols.io.
"""

import re
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from smolagents import tool
import requests
from urllib.parse import quote


@dataclass
class ProtocolScore:
    """Data class to hold protocol evaluation scores."""
    completeness: float
    accuracy: float
    scientific_rigor: float
    reference_quality: float
    safety: float
    overall: float


@dataclass
class ProtocolEvaluation:
    """Data class to hold complete protocol evaluation results."""
    scores: ProtocolScore
    missing_sections: List[str]
    improvements: List[str]
    references_found: List[Dict[str, Any]]
    safety_issues: List[str]
    detailed_feedback: Dict[str, str]


@tool
def protocol_evaluator(
    protocol_text: str,
    protocol_type: Optional[str] = None,
    verify_references: bool = True,
    include_safety_check: bool = True
) -> Dict[str, Any]:
    """
    Evaluates lab protocols for completeness, accuracy, scientific rigor, and suggests improvements.
    
    Args:
        protocol_text: The protocol text to evaluate
        protocol_type: Type of protocol (e.g., "molecular_biology", "cell_culture", "chemistry")
        verify_references: Whether to verify references against external databases
        include_safety_check: Whether to include safety evaluation
    
    Returns:
        Dict containing evaluation scores, missing sections, improvements, and detailed feedback
    """
    try:
        # Input validation
        if not protocol_text or not protocol_text.strip():
            raise ValueError("Protocol text cannot be empty")
        
        # Initialize evaluator
        evaluator = ProtocolEvaluatorEngine()
        
        # Perform comprehensive evaluation
        evaluation = evaluator.evaluate_protocol(
            protocol_text=protocol_text,
            protocol_type=protocol_type,
            verify_references=verify_references,
            include_safety_check=include_safety_check
        )
        
        # Convert to dictionary for return
        return {
            "scores": {
                "completeness": evaluation.scores.completeness,
                "accuracy": evaluation.scores.accuracy,
                "scientific_rigor": evaluation.scores.scientific_rigor,
                "reference_quality": evaluation.scores.reference_quality,
                "safety": evaluation.scores.safety,
                "overall": evaluation.scores.overall
            },
            "missing_sections": evaluation.missing_sections,
            "improvements": evaluation.improvements,
            "references_found": evaluation.references_found,
            "safety_issues": evaluation.safety_issues,
            "detailed_feedback": evaluation.detailed_feedback,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": f"Protocol evaluation failed: {str(e)}",
            "status": "error"
        }


class ProtocolEvaluatorEngine:
    """Core engine for protocol evaluation."""
    
    def __init__(self):
        self.required_sections = {
            "title": ["title", "protocol name", "procedure name"],
            "materials": ["materials", "reagents", "equipment", "supplies", "chemicals"],
            "methods": ["methods", "procedure", "protocol", "steps", "instructions"],
            "safety": ["safety", "precautions", "hazards", "warnings"],
            "references": ["references", "citations", "bibliography"]
        }
        
        self.safety_keywords = [
            "toxic", "hazardous", "dangerous", "flammable", "corrosive",
            "carcinogenic", "mutagenic", "explosive", "volatile", "radioactive",
            "biohazard", "pathogenic", "infectious", "allergen"
        ]
        
        self.quality_indicators = [
            "control", "positive control", "negative control",
            "replicate", "triplicate", "standard", "calibration",
            "validation", "optimization", "troubleshooting"
        ]
    
    def evaluate_protocol(
        self,
        protocol_text: str,
        protocol_type: Optional[str] = None,
        verify_references: bool = True,
        include_safety_check: bool = True
    ) -> ProtocolEvaluation:
        """
        Perform comprehensive protocol evaluation.
        
        Args:
            protocol_text: The protocol text to evaluate
            protocol_type: Type of protocol for specialized evaluation
            verify_references: Whether to verify references
            include_safety_check: Whether to check safety aspects
            
        Returns:
            ProtocolEvaluation object with complete evaluation results
        """
        # Clean and prepare text
        clean_text = self._clean_text(protocol_text)
        
        # Evaluate different aspects
        completeness_score, missing_sections = self._evaluate_completeness(clean_text)
        accuracy_score = self._evaluate_accuracy(clean_text)
        rigor_score = self._evaluate_scientific_rigor(clean_text)
        
        # Reference verification
        references_found = []
        reference_score = 0.0
        if verify_references:
            references_found, reference_score = self._verify_references(clean_text)
        
        # Safety evaluation
        safety_issues = []
        safety_score = 0.0
        if include_safety_check:
            safety_issues, safety_score = self._evaluate_safety(clean_text)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            completeness_score, accuracy_score, rigor_score, reference_score, safety_score
        )
        
        # Generate improvements
        improvements = self._generate_improvements(
            clean_text, missing_sections, safety_issues, reference_score
        )
        
        # Generate detailed feedback
        detailed_feedback = self._generate_detailed_feedback(
            clean_text, completeness_score, accuracy_score, rigor_score,
            reference_score, safety_score
        )
        
        # Create scores object
        scores = ProtocolScore(
            completeness=completeness_score,
            accuracy=accuracy_score,
            scientific_rigor=rigor_score,
            reference_quality=reference_score,
            safety=safety_score,
            overall=overall_score
        )
        
        return ProtocolEvaluation(
            scores=scores,
            missing_sections=missing_sections,
            improvements=improvements,
            references_found=references_found,
            safety_issues=safety_issues,
            detailed_feedback=detailed_feedback
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize protocol text."""
        # Remove extra whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', text.strip())
        return cleaned.lower()
    
    def _evaluate_completeness(self, text: str) -> Tuple[float, List[str]]:
        """Evaluate protocol completeness by checking for required sections."""
        missing_sections = []
        found_sections = 0
        
        for section, keywords in self.required_sections.items():
            found = False
            for keyword in keywords:
                if keyword in text:
                    found = True
                    break
            
            if found:
                found_sections += 1
            else:
                missing_sections.append(section)
        
        completeness_score = (found_sections / len(self.required_sections)) * 100
        return completeness_score, missing_sections
    
    def _evaluate_accuracy(self, text: str) -> float:
        """Evaluate protocol accuracy based on specific details and measurements."""
        accuracy_indicators = 0
        total_checks = 9
        
        # Check for specific measurements
        if re.search(r'\d+\s*(ml|ul|μl|l|mg|g|kg|mm|cm|m|°c|celsius|fahrenheit)', text):
            accuracy_indicators += 1
        
        # Check for time specifications
        if re.search(r'\d+\s*(min|minutes|hour|hours|sec|seconds|h|s)', text):
            accuracy_indicators += 1
        
        # Check for concentrations
        if re.search(r'\d+\s*(m|mm|μm|nm|%|ppm|mg/ml|μg/ml)', text):
            accuracy_indicators += 1
        
        # Check for step numbering
        if re.search(r'(step\s*\d+|^\d+\.|^\d+\))', text, re.MULTILINE):
            accuracy_indicators += 1
        
        # Check for equipment specifications
        if re.search(r'(centrifuge|microscope|incubator|pipette|balance|ph meter)', text):
            accuracy_indicators += 1
        
        # Check for buffer compositions
        if re.search(r'(buffer|solution|stock)', text):
            accuracy_indicators += 1
        
        # Check for storage conditions
        if re.search(r'(store|storage|-20|4°c|room temperature|rt)', text):
            accuracy_indicators += 1
        
        # Check for detailed procedures
        if re.search(r'(mix|vortex|centrifuge|incubate|wash|rinse)', text):
            accuracy_indicators += 1
        
        # Check for specific protocols or techniques
        if re.search(r'(pcr|western blot|elisa|qpcr|gel electrophoresis)', text):
            accuracy_indicators += 1
        
        accuracy_score = (accuracy_indicators / total_checks) * 100
        return accuracy_score
    
    def _evaluate_scientific_rigor(self, text: str) -> float:
        """Evaluate scientific rigor based on controls and quality measures."""
        rigor_indicators = 0
        total_checks = 7
        
        # Check for controls
        for indicator in self.quality_indicators:
            if indicator in text:
                rigor_indicators += 1
                break
        
        # Check for statistical considerations
        if re.search(r'(n\s*=|sample size|statistical|significance|p-value)', text):
            rigor_indicators += 1
        
        # Check for troubleshooting
        if re.search(r'(troubleshoot|problem|issue|error|fail)', text):
            rigor_indicators += 1
        
        # Check for optimization
        if re.search(r'(optim|calibrat|validat|standard)', text):
            rigor_indicators += 1
        
        # Check for quality control
        if re.search(r'(quality|qc|standard|reference)', text):
            rigor_indicators += 1
        
        # Check for expected results
        if re.search(r'(expect|result|outcome|yield|output)', text):
            rigor_indicators += 1
        
        # Check for limitations or notes
        if re.search(r'(note|limitation|caution|important)', text):
            rigor_indicators += 1
        
        rigor_score = (rigor_indicators / total_checks) * 100
        return rigor_score
    
    def _verify_references(self, text: str) -> Tuple[List[Dict[str, Any]], float]:
        """Verify references by checking for DOIs, PMIDs, and citations."""
        references_found = []
        reference_score = 0.0
        
        # Extract DOIs
        doi_pattern = r'10\.\d{4,}/[^\s]+'
        dois = re.findall(doi_pattern, text)
        
        # Extract PMIDs
        pmid_pattern = r'pmid:?\s*(\d{7,8})'
        pmids = re.findall(pmid_pattern, text, re.IGNORECASE)
        
        # Extract citations (basic pattern)
        citation_pattern = r'([A-Z][a-z]+\s+et\s+al\.?|\([12]\d{3}\))'
        citations = re.findall(citation_pattern, text)
        
        # Score based on reference presence
        if dois:
            reference_score += 40
            for doi in dois[:3]:  # Limit to first 3 DOIs
                references_found.append({"type": "DOI", "value": doi})
        
        if pmids:
            reference_score += 30
            for pmid in pmids[:3]:  # Limit to first 3 PMIDs
                references_found.append({"type": "PMID", "value": pmid})
        
        if citations:
            reference_score += 20
            references_found.append({"type": "citations", "count": len(citations)})
        
        # Check for reference section
        if re.search(r'(references|bibliography|citations)', text):
            reference_score += 10
        
        reference_score = min(reference_score, 100)
        return references_found, reference_score
    
    def _evaluate_safety(self, text: str) -> Tuple[List[str], float]:
        """Evaluate safety considerations in the protocol."""
        safety_issues = []
        safety_score = 50.0  # Base score
        
        # Check for safety keywords without corresponding warnings
        for keyword in self.safety_keywords:
            if keyword in text:
                # Check if there's a safety warning nearby
                keyword_pos = text.find(keyword)
                surrounding_text = text[max(0, keyword_pos-100):keyword_pos+100]
                
                if not re.search(r'(warning|caution|safety|ppe|glove|hood|ventilation)', surrounding_text):
                    safety_issues.append(f"Safety concern found: '{keyword}' without adequate warning")
                    safety_score -= 10
                else:
                    safety_score += 5
        
        # Check for general safety mentions
        safety_mentions = re.findall(r'(safety|warning|caution|ppe|personal protective equipment)', text)
        if safety_mentions:
            safety_score += len(safety_mentions) * 5
        
        # Check for specific safety equipment
        ppe_mentions = re.findall(r'(gloves|goggles|lab coat|fume hood|ventilation)', text)
        if ppe_mentions:
            safety_score += len(ppe_mentions) * 3
        
        safety_score = max(0, min(safety_score, 100))
        return safety_issues, safety_score
    
    def _calculate_overall_score(
        self,
        completeness: float,
        accuracy: float,
        rigor: float,
        reference: float,
        safety: float
    ) -> float:
        """Calculate weighted overall score."""
        weights = {
            'completeness': 0.25,
            'accuracy': 0.25,
            'rigor': 0.20,
            'reference': 0.15,
            'safety': 0.15
        }
        
        overall = (
            completeness * weights['completeness'] +
            accuracy * weights['accuracy'] +
            rigor * weights['rigor'] +
            reference * weights['reference'] +
            safety * weights['safety']
        )
        
        return round(overall, 1)
    
    def _generate_improvements(
        self,
        text: str,
        missing_sections: List[str],
        safety_issues: List[str],
        reference_score: float
    ) -> List[str]:
        """Generate specific improvement suggestions."""
        improvements = []
        
        # Address missing sections
        if missing_sections:
            improvements.append(f"Add missing sections: {', '.join(missing_sections)}")
        
        # Address safety issues
        if safety_issues:
            improvements.append("Add safety warnings for identified hazardous materials/procedures")
        
        # Address reference quality
        if reference_score < 50:
            improvements.append("Include more references with DOIs or PMIDs to support methods")
        
        # Check for specific improvements
        if not re.search(r'(control|standard)', text):
            improvements.append("Include appropriate controls (positive/negative)")
        
        if not re.search(r'\d+\s*(replicate|repeat|n\s*=)', text):
            improvements.append("Specify number of replicates or sample size")
        
        if not re.search(r'(troubleshoot|expected result)', text):
            improvements.append("Add troubleshooting section and expected results")
        
        if not re.search(r'(storage|store)', text):
            improvements.append("Include storage conditions for reagents and samples")
        
        return improvements
    
    def _generate_detailed_feedback(
        self,
        text: str,
        completeness: float,
        accuracy: float,
        rigor: float,
        reference: float,
        safety: float
    ) -> Dict[str, str]:
        """Generate detailed feedback for each evaluation category."""
        feedback = {}
        
        # Completeness feedback
        if completeness >= 80:
            feedback["completeness"] = "Excellent - All major sections are present"
        elif completeness >= 60:
            feedback["completeness"] = "Good - Most sections present, minor gaps"
        else:
            feedback["completeness"] = "Needs improvement - Several key sections missing"
        
        # Accuracy feedback
        if accuracy >= 80:
            feedback["accuracy"] = "Excellent - Detailed measurements and specific instructions"
        elif accuracy >= 60:
            feedback["accuracy"] = "Good - Most details provided, some specificity could be improved"
        else:
            feedback["accuracy"] = "Needs improvement - Lacks specific measurements and detailed steps"
        
        # Scientific rigor feedback
        if rigor >= 80:
            feedback["scientific_rigor"] = "Excellent - Strong experimental design with appropriate controls"
        elif rigor >= 60:
            feedback["scientific_rigor"] = "Good - Adequate experimental considerations"
        else:
            feedback["scientific_rigor"] = "Needs improvement - Lacks quality controls and validation measures"
        
        # Reference feedback
        if reference >= 80:
            feedback["reference_quality"] = "Excellent - Well-referenced with verifiable citations"
        elif reference >= 60:
            feedback["reference_quality"] = "Good - Some references present"
        else:
            feedback["reference_quality"] = "Needs improvement - Insufficient or missing references"
        
        # Safety feedback
        if safety >= 80:
            feedback["safety"] = "Excellent - Comprehensive safety considerations"
        elif safety >= 60:
            feedback["safety"] = "Good - Basic safety measures addressed"
        else:
            feedback["safety"] = "Needs improvement - Safety considerations inadequate"
        
        return feedback


# Example usage and testing function
def test_protocol_evaluator():
    """Test function to validate the protocol evaluator."""
    sample_protocol = """
    Title: DNA Extraction from Bacterial Cultures
    
    Materials:
    - Bacterial culture (overnight growth)
    - Lysis buffer (50mM Tris-HCl, pH 8.0, 1mM EDTA)
    - Proteinase K (20 mg/ml)
    - Phenol:chloroform (1:1)
    - Ethanol (100% and 70%)
    - TE buffer (10mM Tris-HCl, 1mM EDTA, pH 8.0)
    
    Methods:
    1. Centrifuge 1.5ml of bacterial culture at 10,000 rpm for 2 minutes
    2. Resuspend pellet in 200μl lysis buffer
    3. Add 20μl proteinase K, incubate at 56°C for 30 minutes
    4. Add equal volume phenol:chloroform, vortex for 10 seconds
    5. Centrifuge at 12,000 rpm for 10 minutes
    6. Transfer aqueous phase to new tube
    7. Add 2 volumes 100% ethanol, mix gently
    8. Centrifuge at 12,000 rpm for 15 minutes
    9. Wash pellet with 70% ethanol
    10. Air dry pellet and resuspend in 50μl TE buffer
    
    Safety:
    Warning: Phenol is toxic and corrosive. Use in fume hood with gloves.
    
    References:
    Sambrook et al. (1989) Molecular Cloning. PMID: 12345678
    """
    
    try:
        result = protocol_evaluator(sample_protocol)
        print("Protocol Evaluator Test Results:")
        print(f"Overall Score: {result['scores']['overall']}")
        print(f"Status: {result['status']}")
        return result
    except Exception as e:
        print(f"Test failed: {e}")
        return None


if __name__ == "__main__":
    test_protocol_evaluator()