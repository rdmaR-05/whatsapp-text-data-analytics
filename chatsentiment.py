import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from textblob import TextBlob
import re
import os

plt.style.use('ggplot')
sns.set_palette("viridis")

# 1. Parse the wpchat
def parse_chat(file_path):
    pattern = r"\[(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{2}:\d{2})\s?(AM|PM)?\] (.*?): (.*)"
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.match(pattern, line)
            if match:
                date_str, time_str, ampm, sender, message = match.groups()
                datetime_str = f"{date_str} {time_str} {ampm}" if ampm else f"{date_str} {time_str}"
                try:
                    dt = datetime.strptime(datetime_str, "%m/%d/%y %I:%M:%S %p") if ampm else datetime.strptime(datetime_str, "%m/%d/%y %H:%M:%S")
                    data.append((dt, sender.strip(), message.strip(), os.path.basename(file_path)))
                except Exception as e:
                    continue
    df = pd.DataFrame(data, columns=["datetime", "sender", "message", "chat_file"])
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    df['period'] = df['hour'] // 2 * 2
    df['sentiment'] = df['message'].apply(lambda x: TextBlob(x).sentiment.polarity)
    return df

# 2. Combine multiple chats
def load_all_chats(folder_path):
    all_data = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            file_path = os.path.join(folder_path, file)
            df = parse_chat(file_path)
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

# 3. Date Activity Chart
def create_date_activity_chart(df, title_suffix=""):
    plt.figure(figsize=(12, 8))
    date_counts = df.groupby('date').size()
    plt.plot(date_counts.index, date_counts.values, 'o-', linewidth=2, markersize=6)
    plt.title(f'Message Activity Over Time {title_suffix}', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Number of Messages')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'fig1_date_activity{title_suffix}.png')
    plt.show()

# 4. Hourly Activity Chart
def create_hourly_activity_chart(df, title_suffix=""):
    plt.figure(figsize=(12, 8))
    hour_counts = df['hour'].value_counts().sort_index()
    plt.bar(hour_counts.index, hour_counts.values, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f'Message Activity by Hour {title_suffix}')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Messages')
    plt.xticks(range(0, 24, 2), [f'{h}:00' for h in range(0, 24, 2)])
    plt.tight_layout()
    plt.savefig(f'fig2_hourly_activity{title_suffix}.png')
    plt.show()

# 5. Period Activity Chart
def create_period_activity_chart(df, title_suffix=""):
    plt.figure(figsize=(12, 8))
    period_counts = df['period'].value_counts().sort_index()
    labels = [f"{p}-{p+2}:00" for p in period_counts.index]
    plt.bar(labels, period_counts.values, color='lightgreen', edgecolor='black', alpha=0.7)
    plt.title(f'Message Activity by 2-Hour Periods {title_suffix}')
    plt.xlabel('Time Period')
    plt.ylabel('Number of Messages')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'fig3_period_activity{title_suffix}.png')
    plt.show()

# 6. User Sentiment Chart
def create_user_sentiment_chart(df, title_suffix=""):
    plt.figure(figsize=(12, 8))
    sentiment_by_user = df.groupby('sender')['sentiment'].mean().sort_values()
    y_pos = np.arange(len(sentiment_by_user))
    plt.barh(y_pos, sentiment_by_user.values, color='orange', alpha=0.7)
    plt.yticks(y_pos, sentiment_by_user.index)
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel('Sentiment Score')
    plt.title(f'Sentiment Analysis by User {title_suffix}')
    plt.tight_layout()
    plt.savefig(f'fig4_user_sentiment{title_suffix}.png')
    plt.show()

# 7. Sentiment Distribution Chart
def create_sentiment_distribution_chart(df, title_suffix=""):
    fig, ax = plt.subplots(figsize=(12, 8))
    sentiments = df['sentiment']
    ax.hist(sentiments, bins=40, range=(-1, 1), color='blue', alpha=0.6, edgecolor='black')
    mean_sentiment = sentiments.mean()
    ax.axvline(mean_sentiment, color='red', linestyle='--', linewidth=2)
    ax.set_title(f'Sentiment Distribution {title_suffix}')
    ax.set_xlabel('Sentiment Score')
    ax.set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'fig5_sentiment_distribution{title_suffix}.png')
    plt.show()

# 8. Main
if __name__ == "__main__":
    folder = "chats" 
    all_chats = load_all_chats(folder)

    for chat_file in all_chats['chat_file'].unique():
        print(f"\nAnalyzing chat: {chat_file}")
        chat_df = all_chats[all_chats['chat_file'] == chat_file]
        suffix = f"_{chat_file.replace('.txt','')}"
        create_date_activity_chart(chat_df, suffix)
        create_hourly_activity_chart(chat_df, suffix)
        create_period_activity_chart(chat_df, suffix)
        create_user_sentiment_chart(chat_df, suffix)
        create_sentiment_distribution_chart(chat_df, suffix)
    print("\nAnalyzing combined chats")
    create_date_activity_chart(all_chats, "_combined")
    create_hourly_activity_chart(all_chats, "_combined")
    create_period_activity_chart(all_chats, "_combined")
    print("\nAll charts created!")
