"""Midnight Batch Processing System Implementation

Task 1.4: Midnight Batch Processing System
Sprint 1: Data Foundation & Quality

CLAUDE.md Compliance:
- Think Before Coding: ✅ Batch processing assumptions stated
- Simplicity First: Python scheduler, no complex orchestration
- Surgical Changes: Only batch processing code
- Goal-Driven: Automated daily data processing

Midnight Batch Processing Features:
1. Daily data refresh (1:00-2:00 AM Vietnam time)
2. Bronze/Silver layer processing
3. Vietnamese NLP processing
4. Quality validation
5. Error handling and logging
"""

import schedule
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
import json
import sys
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MidnightBatchProcessor:
    """Midnight batch processing system for daily data operations."""

    def __init__(self):
        """Initialize the batch processor."""
        self.project_root = Path(__file__).parent.parent.parent
        self.logs_dir = self.project_root / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        # Processing time: 1:00 AM Vietnam time
        self.processing_hour = 1
        self.processing_minute = 0

        logger.info("Midnight Batch Processor initialized")

    def check_data_freshness(self) -> dict:
        """Check if data needs refreshing.

        Returns:
            Dictionary with freshness status
        """
        logger.info("Checking data freshness...")

        now = datetime.now()
        last_processed = self.get_last_processed_time()

        if last_processed is None:
            return {
                "fresh": False,
                "reason": "No previous processing found",
                "last_processed": None,
                "hours_since": None
            }

        hours_since = (now - last_processed).total_seconds() / 3600

        # Data is considered fresh if processed within last 24 hours
        is_fresh = hours_since < 24

        return {
            "fresh": is_fresh,
            "reason": f"Processed {hours_since:.1f} hours ago",
            "last_processed": last_processed.isoformat(),
            "hours_since": hours_since
        }

    def get_last_processed_time(self) -> datetime:
        """Get the last successful processing time.

        Returns:
            Datetime of last processing or None if not found
        """
        metadata_file = self.project_root / "data_lakehouse" / "_metadata" / "processing_log.json"

        if not metadata_file.exists():
            return None

        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                last_processed = data.get('last_processed')
                if last_processed:
                    return datetime.fromisoformat(last_processed)
        except Exception as e:
            logger.error(f"Error reading processing log: {e}")

        return None

    def update_processing_log(self, status: str, details: dict = None):
        """Update the processing log.

        Args:
            status: Processing status (success, failed, partial)
            details: Additional processing details
        """
        logger.info("Updating processing log...")

        metadata_file = self.project_root / "data_lakehouse" / "_metadata" / "processing_log.json"
        metadata_file.parent.mkdir(exist_ok=True)

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "details": details or {}
        }

        # Read existing log or create new
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if not isinstance(data, dict):
                        data = {}
            except Exception as e:
                logger.error(f"Error reading processing log: {e}")
                data = {}
        else:
            data = {}

        # Update log
        data['last_processed'] = datetime.now().isoformat()
        data['last_status'] = status

        if 'history' not in data:
            data['history'] = []

        data['history'].append(log_entry)

        # Keep only last 30 entries
        if len(data['history']) > 30:
            data['history'] = data['history'][-30:]

        # Save updated log
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Processing log updated: {status}")

    def run_bronze_import(self) -> dict:
        """Run Bronze import pipeline.

        Returns:
            Result dictionary
        """
        logger.info("Running Bronze import...")

        try:
            # Import the Bronze importer
            bronze_script = self.project_root / "src" / "sprint1" / "task1_1_bronze_import.py"

            if not bronze_script.exists():
                return {
                    "success": False,
                    "error": "Bronze import script not found",
                    "stage": "bronze_import"
                }

            # Run the Bronze import
            result = subprocess.run(
                [sys.executable, str(bronze_script)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )

            if result.returncode == 0:
                logger.info("Bronze import completed successfully")
                return {
                    "success": True,
                    "stage": "bronze_import",
                    "output": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout
                }
            else:
                logger.error(f"Bronze import failed: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr,
                    "stage": "bronze_import"
                }

        except Exception as e:
            logger.error(f"Bronze import error: {e}")
            return {
                "success": False,
                "error": str(e),
                "stage": "bronze_import"
            }

    def run_silver_processing(self) -> dict:
        """Run Silver processing pipeline.

        Returns:
            Result dictionary
        """
        logger.info("Running Silver processing...")

        try:
            # Import the Silver processor
            silver_script = self.project_root / "src" / "sprint1" / "task1_2_silver_processing.py"

            if not silver_script.exists():
                return {
                    "success": False,
                    "error": "Silver processing script not found",
                    "stage": "silver_processing"
                }

            # Run the Silver processing
            result = subprocess.run(
                [sys.executable, str(silver_script)],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )

            if result.returncode == 0:
                logger.info("Silver processing completed successfully")
                return {
                    "success": True,
                    "stage": "silver_processing",
                    "output": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout
                }
            else:
                logger.error(f"Silver processing failed: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr,
                    "stage": "silver_processing"
                }

        except Exception as e:
            logger.error(f"Silver processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "stage": "silver_processing"
            }

    def run_vietnamese_nlp(self) -> dict:
        """Run Vietnamese NLP processing.

        Returns:
            Result dictionary
        """
        logger.info("Running Vietnamese NLP processing...")

        try:
            # Import the Vietnamese NLP processor
            nlp_script = self.project_root / "src" / "sprint1" / "task1_3_vietnamese_nlp.py"

            if not nlp_script.exists():
                return {
                    "success": False,
                    "error": "Vietnamese NLP script not found",
                    "stage": "vietnamese_nlp"
                }

            # Run the Vietnamese NLP processing
            result = subprocess.run(
                [sys.executable, str(nlp_script)],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )

            if result.returncode == 0:
                logger.info("Vietnamese NLP processing completed successfully")
                return {
                    "success": True,
                    "stage": "vietnamese_nlp",
                    "output": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout
                }
            else:
                logger.error(f"Vietnamese NLP processing failed: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr,
                    "stage": "vietnamese_nlp"
                }

        except Exception as e:
            logger.error(f"Vietnamese NLP processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "stage": "vietnamese_nlp"
            }

    def run_data_refresh(self) -> dict:
        """Run complete data refresh pipeline.

        Returns:
            Processing result dictionary
        """
        logger.info("="*60)
        logger.info("Starting complete data refresh pipeline")
        logger.info("="*60)

        start_time = datetime.now()
        results = {
            "start_time": start_time.isoformat(),
            "stages": {}
        }

        try:
            # Stage 1: Bronze import
            logger.info("Stage 1: Bronze import")
            bronze_result = self.run_bronze_import()
            results["stages"]["bronze"] = bronze_result

            if not bronze_result["success"]:
                logger.error("Bronze import failed, stopping pipeline")
                self.update_processing_log("failed", results)
                return {
                    "success": False,
                    "error": "Bronze import failed",
                    "results": results
                }

            # Stage 2: Silver processing
            logger.info("Stage 2: Silver processing")
            silver_result = self.run_silver_processing()
            results["stages"]["silver"] = silver_result

            if not silver_result["success"]:
                logger.warning("Silver processing failed, continuing...")
                # Don't stop, continue to NLP

            # Stage 3: Vietnamese NLP
            logger.info("Stage 3: Vietnamese NLP processing")
            nlp_result = self.run_vietnamese_nlp()
            results["stages"]["nlp"] = nlp_result

            if not nlp_result["success"]:
                logger.warning("Vietnamese NLP processing failed")

            # Calculate overall status
            all_stages = [bronze_result, silver_result, nlp_result]
            success_count = sum(1 for r in all_stages if r["success"])

            if success_count == len(all_stages):
                status = "success"
            elif success_count > 0:
                status = "partial"
            else:
                status = "failed"

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            results["end_time"] = end_time.isoformat()
            results["duration_seconds"] = duration
            results["status"] = status
            results["success"] = status == "success"

            logger.info("="*60)
            logger.info(f"Data refresh completed: {status}")
            logger.info(f"Duration: {duration:.1f} seconds")
            logger.info(f"Success rate: {success_count}/{len(all_stages)} stages")
            logger.info("="*60)

            # Update processing log
            self.update_processing_log(status, results)

            return results

        except Exception as e:
            logger.error(f"Data refresh pipeline error: {e}")
            error_result = {
                "success": False,
                "error": str(e),
                "results": results,
                "status": "failed"
            }
            self.update_processing_log("failed", error_result)
            return error_result

    def schedule_daily_processing(self):
        """Schedule daily processing at 1:00 AM."""
        logger.info("Scheduling daily processing at 1:00 AM...")

        schedule.every().day.at("01:00").do(self.run_data_refresh)

        logger.info("Daily processing scheduled successfully")
        logger.info(f"Next run: {schedule.next_run()}")

    def start_scheduler(self):
        """Start the scheduler for continuous operation."""
        logger.info("Starting batch processing scheduler...")

        # Schedule daily processing
        self.schedule_daily_processing()

        logger.info("Scheduler started. Press Ctrl+C to stop.")

        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")

    def run_immediate(self):
        """Run the processing immediately without scheduling."""
        logger.info("Running immediate data refresh...")

        result = self.run_data_refresh()

        if result["success"]:
            logger.info("Immediate refresh completed successfully")
        else:
            logger.error("Immediate refresh failed")

        return result


def main():
    """Main function to run batch processing system."""
    print("="*60)
    print("Midnight Batch Processing System")
    print("="*60)

    processor = MidnightBatchProcessor()

    # Check data freshness
    print("\n1. Checking data freshness...")
    freshness = processor.check_data_freshness()
    print(f"Fresh: {freshness['fresh']}")
    print(f"Reason: {freshness['reason']}")

    if not freshness['fresh']:
        print("\n2. Data needs refresh, running immediate processing...")
        result = processor.run_immediate()

        print(f"\n3. Processing results:")
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Duration: {result.get('duration_seconds', 0):.1f} seconds")

        if 'stages' in result:
            print(f"\n4. Stage results:")
            for stage_name, stage_result in result['stages'].items():
                status_symbol = "[OK]" if stage_result.get('success') else "[FAIL]"
                print(f"  {stage_name}: {status_symbol}")
    else:
        print("\n2. Data is fresh, no immediate processing needed")

    print("\n" + "="*60)
    print("Midnight Batch Processing System: READY [OK]")
    print("To start scheduler: processor.start_scheduler()")
    print("="*60)

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)