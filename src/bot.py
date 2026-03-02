import discord
from discord import app_commands
import logging
from src.agent.core import DiscordAgentGraph

logger = logging.getLogger('discord_bot')

class AIAgentBot(discord.Client):
    def __init__(self):
        # 1. 设置 Intents (Intents)
        # 接收私信文字内容需要 message_content (message_content is needed to read message bodies)
        intents = discord.Intents.default()
        intents.message_content = True 
        super().__init__(intents=intents)
        
        # 2. 创建指令树 (Command tree)
        self.tree = app_commands.CommandTree(self)
        
        # Instantiate the agent graph
        self.agent_graph = DiscordAgentGraph()

    async def setup_hook(self):
        # 启动时同步指令到 Discord (Sync commands on startup)
        await self.tree.sync()
        logger.info(f"Synced slash commands for {self.user}")

    async def on_ready(self):
        logger.info(f'Logged in as {self.user} (ID: {self.user.id})')
        logger.info('------')

    async def on_message(self, message: discord.Message):
        # 排除机器人自己的消息 (Ignore bot's own messages)
        if message.author == self.user:
            return

        # 检查是否是私信 (Check if it's a Direct Message)
        if isinstance(message.channel, discord.DMChannel):
            user_input = message.content
            logger.info(f"Received DM from {message.author}: {user_input}")
            
            async with message.channel.typing():
                response = await self.agent_graph.chat(
                    user_id=str(message.author.id),
                    user_name=message.author.display_name,
                    user_input=user_input
                )
                
                # Discord 限制单次消息为2000字符 (Discord limits messages to 2000 characters)
                if len(response) > 2000:
                    for i in range(0, len(response), 2000):
                        await message.channel.send(response[i:i+2000])
                else:
                    await message.channel.send(response)

def setup_bot():
    bot = AIAgentBot()

    # Define integration capabilities if available in installed discord.py version
    try:
        integration_types = {
            app_commands.AppInstallationType.guild,
            app_commands.AppInstallationType.user
        }
        contexts = {
            app_commands.AppCommandContext.guild,
            app_commands.AppCommandContext.bot_dm,
            app_commands.AppCommandContext.private_channel
        }
    except AttributeError:
        # Fallback for older discord.py versions
        integration_types = None
        contexts = None

    command_kwargs = {
        "name": "chat",
        "description": "与我对话 (Chat with me)"
    }
    
    if integration_types:
        command_kwargs["integration_types"] = integration_types
    if contexts:
        command_kwargs["contexts"] = contexts

    # --- 场景 A: 用户通过 /chat 指令发起对话 ---
    @bot.tree.command(**command_kwargs)
    async def chat(interaction: discord.Interaction):
        # Verify if executed in guild or direct message
        if interaction.guild_id is not None:
            await interaction.response.send_message("你好！为了隐私，请在私信(DM)中直接跟我说话。", ephemeral=True)
        else:
            await interaction.response.send_message("你好！我是你的私人AI助手，请直接发送消息与我对话。")

    return bot
