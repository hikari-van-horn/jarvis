import os
import sys
import logging
import asyncio
import importlib

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('main')

# Ensure the root folder is added to Python path for src/ imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def run_extensions(ext_names):
    tasks = []
    
    for ext_name in ext_names:
        module_name = f"src.exts.{ext_name}_ext"
        try:
            ext_module = importlib.import_module(module_name)
            if hasattr(ext_module, "start") and asyncio.iscoroutinefunction(ext_module.start):
                logger.info(f"Starting extension: {ext_name}")
                task = asyncio.create_task(ext_module.start())
                tasks.append(task)
            else:
                logger.error(f"Extension {ext_name} does not have an async 'start' function. Skipping.")
        except ImportError as e:
            logger.error(f"Could not load extension '{ext_name}': {e}")
        except Exception as e:
            logger.error(f"Error initializing extension '{ext_name}': {e}")

    if not tasks:
        logger.error("No valid extensions to run. Exiting.")
        return

    logger.info(f"Running {len(tasks)} extensions concurrently...")
    # Gather tasks and run them indefinitely until completion or Exception
    await asyncio.gather(*tasks)

def main():
    # Parse command line parameters looking for exts=...
    ext_names = ["discord"] # Check default
    for arg in sys.argv[1:]:
        if arg.startswith("exts="):
            exts_str = arg[len("exts="):]
            ext_names = [e.strip() for e in exts_str.split(",") if e.strip()]
            
    if not ext_names:
        logger.error("No extensions provided. Try running with: exts=discord,telegram")
        sys.exit(1)
        
    try:
        asyncio.run(run_extensions(ext_names))
    except KeyboardInterrupt:
        logger.info("Application shut down by user.")

if __name__ == "__main__":
    main()
