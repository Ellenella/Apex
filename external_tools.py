import os
import json
import csv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ExternalTools:
    def __init__(self):
        """Initialize external tools with configuration."""
        self.logger = logging.getLogger(__name__)
        
        # Slack configuration
        self.slack_token = os.getenv('SLACK_BOT_TOKEN')
        self.slack_channel = os.getenv('SLACK_CHANNEL_ID')
        
        # Email configuration
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', 587))
        self.email_user = os.getenv('EMAIL_USER')
        self.email_password = os.getenv('EMAIL_PASSWORD')
        
        # Google Sheets configuration
        self.google_credentials_file = os.getenv('GOOGLE_SHEETS_CREDENTIALS_FILE')
    
    def send_slack_notification(self, message, recommendations=None, priority='normal'):
        """Send notification to Slack channel."""
        try:
            if not self.slack_token or not self.slack_channel:
                self.logger.warning("Slack credentials not configured")
                return False
            
            # Import slack_sdk here to avoid dependency issues
            try:
                from slack_sdk import WebClient
                from slack_sdk.errors import SlackApiError
            except ImportError:
                self.logger.error("slack_sdk not installed. Install with: pip install slack-sdk")
                return False
            
            client = WebClient(token=self.slack_token)
            
            # Create message blocks
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"🚀 Procurement Recommendation Alert ({priority.upper()})"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": message
                    }
                }
            ]
            
            # Add recommendations if provided
            if recommendations and len(recommendations) > 0:
                top_rec = recommendations[0]
                blocks.append({
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Top Supplier:*\n{top_rec['supplier_name']}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Score:*\n{top_rec['supplier_score']:.2f}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Total Cost:*\n${top_rec['total_cost']:.2f}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Risk Score:*\n{top_rec['risk_score']:.2f}"
                        }
                    ]
                })
            
            # Add timestamp
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    }
                ]
            })
            
            # Send message
            response = client.chat_postMessage(
                channel=self.slack_channel,
                blocks=blocks,
                text=message
            )
            
            self.logger.info(f"Slack notification sent successfully: {response['ts']}")
            return True
            
        except SlackApiError as e:
            self.logger.error(f"Slack API error: {e.response['error']}")
            return False
        except Exception as e:
            self.logger.error(f"Error sending Slack notification: {e}")
            return False
    
    def send_email_notification(self, subject, message, recipients, recommendations=None):
        """Send email notification with procurement recommendations."""
        try:
            if not self.email_user or not self.email_password:
                self.logger.warning("Email credentials not configured")
                return False
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_user
            msg['To'] = ', '.join(recipients) if isinstance(recipients, list) else recipients
            
            # Create HTML content
            html_content = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
                    .recommendation {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
                    .score {{ font-weight: bold; color: #28a745; }}
                    .risk {{ font-weight: bold; color: #dc3545; }}
                    .cost {{ font-weight: bold; color: #007bff; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h2>📋 Procurement Recommendation Report</h2>
                    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div>
                    <h3>Summary</h3>
                    <p>{message}</p>
                </div>
            """
            
            # Add recommendations if provided
            if recommendations and len(recommendations) > 0:
                html_content += "<h3>Top Recommendations</h3>"
                
                for i, rec in enumerate(recommendations[:3], 1):
                    html_content += f"""
                    <div class="recommendation">
                        <h4>{i}. {rec['supplier_name']}</h4>
                        <p><strong>Category:</strong> {rec['category']}</p>
                        <p><strong>Location:</strong> {rec['location']}</p>
                        <p><strong>Lead Time:</strong> {rec['lead_time_days']} days</p>
                        <p><strong>Unit Cost:</strong> ${rec['unit_cost']:.2f}</p>
                        <p><strong>Recommended Quantity:</strong> {rec['recommended_quantity']}</p>
                        <p><strong>Total Cost:</strong> <span class="cost">${rec['total_cost']:.2f}</span></p>
                        <p><strong>Supplier Score:</strong> <span class="score">{rec['supplier_score']:.2f}</span></p>
                        <p><strong>Risk Score:</strong> <span class="risk">{rec['risk_score']:.2f}</span></p>
                        <p><strong>On-Time Delivery:</strong> {rec['on_time_delivery_rate']:.1%}</p>
                        <p><strong>Quality Score:</strong> {rec['quality_score']:.1f}/5.0</p>
                    </div>
                    """
            
            html_content += """
                <div style="margin-top: 30px; padding: 15px; background-color: #f8f9fa; border-radius: 5px;">
                    <p><em>This is an automated message from the Procurement Optimization System.</em></p>
                </div>
            </body>
            </html>
            """
            
            # Attach HTML content
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_password)
                server.send_message(msg)
            
            self.logger.info(f"Email notification sent to {recipients}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending email notification: {e}")
            return False
    
    def export_to_google_sheets(self, data, sheet_name="Procurement_Recommendations"):
        """Export recommendations to Google Sheets."""
        try:
            if not self.google_credentials_file:
                self.logger.warning("Google Sheets credentials not configured")
                return False
            
            # Import gspread here to avoid dependency issues
            try:
                import gspread
                from google.oauth2.service_account import Credentials
            except ImportError:
                self.logger.error("gspread not installed. Install with: pip install gspread google-auth")
                return False
            
            # Setup credentials
            scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
            credentials = Credentials.from_service_account_file(self.google_credentials_file, scopes=scope)
            client = gspread.authorize(credentials)
            
            # Create or open spreadsheet
            try:
                spreadsheet = client.open(sheet_name)
            except gspread.SpreadsheetNotFound:
                spreadsheet = client.create(sheet_name)
                self.logger.info(f"Created new Google Sheet: {sheet_name}")
            
            # Prepare data for export
            if isinstance(data, list) and len(data) > 0:
                # Export recommendations
                headers = [
                    'Supplier ID', 'Supplier Name', 'Category', 'Rating', 'Lead Time (days)',
                    'Unit Cost', 'Recommended Quantity', 'Total Cost', 'Supplier Score',
                    'Risk Score', 'On-Time Delivery Rate', 'Quality Score', 'Location',
                    'Payment Terms', 'Export Date'
                ]
                
                rows = [headers]
                for rec in data:
                    row = [
                        rec.get('supplier_id', ''),
                        rec.get('supplier_name', ''),
                        rec.get('category', ''),
                        rec.get('rating', ''),
                        rec.get('lead_time_days', ''),
                        rec.get('unit_cost', ''),
                        rec.get('recommended_quantity', ''),
                        rec.get('total_cost', ''),
                        rec.get('supplier_score', ''),
                        rec.get('risk_score', ''),
                        rec.get('on_time_delivery_rate', ''),
                        rec.get('quality_score', ''),
                        rec.get('location', ''),
                        rec.get('payment_terms', ''),
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ]
                    rows.append(row)
                
                # Create worksheet
                worksheet_name = f"Recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=len(rows), cols=len(headers))
                
                # Update data
                worksheet.update(rows)
                
                self.logger.info(f"Data exported to Google Sheets: {spreadsheet.url}")
                return spreadsheet.url
            
            else:
                self.logger.warning("No data to export")
                return False
                
        except Exception as e:
            self.logger.error(f"Error exporting to Google Sheets: {e}")
            return False
    
    def export_to_csv(self, data, filename=None):
        """Export recommendations to CSV file."""
        try:
            if not data or len(data) == 0:
                self.logger.warning("No data to export")
                return False
            
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"procurement_recommendations_{timestamp}.csv"
            
            # Define headers
            headers = [
                'supplier_id', 'supplier_name', 'category', 'rating', 'lead_time_days',
                'unit_cost', 'recommended_quantity', 'total_cost', 'supplier_score',
                'risk_score', 'on_time_delivery_rate', 'quality_score', 'location',
                'payment_terms', 'export_date'
            ]
            
            # Write to CSV
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                
                for rec in data:
                    # Add export date
                    rec['export_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    writer.writerow(rec)
            
            self.logger.info(f"Data exported to CSV: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {e}")
            return False
    
    def send_procurement_alert(self, recommendations, market_analysis, alert_type='recommendation'):
        """Send comprehensive procurement alert with all available channels."""
        try:
            # Create alert message
            if alert_type == 'recommendation':
                subject = "🚀 New Procurement Recommendations Available"
                message = f"Found {len(recommendations)} supplier recommendations for your procurement needs."
                
                if market_analysis:
                    message += f"\n\nMarket Analysis:\n"
                    message += f"- Total spent: ${market_analysis.get('total_spent', 0):,.2f}\n"
                    message += f"- Cost trend: {market_analysis.get('cost_trend', 'stable')}\n"
                    message += f"- Demand trend: {market_analysis.get('demand_trend', 'stable')}"
            
            elif alert_type == 'urgent':
                subject = "⚠️ URGENT: Procurement Action Required"
                message = "High-priority procurement recommendations require immediate attention."
            
            else:
                subject = "📊 Procurement Analysis Complete"
                message = "Procurement analysis has been completed successfully."
            
            # Send notifications through all available channels
            results = {
                'slack': False,
                'email': False,
                'google_sheets': False,
                'csv': False
            }
            
            # Slack notification
            if self.slack_token and self.slack_channel:
                results['slack'] = self.send_slack_notification(
                    message, recommendations, 
                    priority='high' if alert_type == 'urgent' else 'normal'
                )
            
            # Email notification (if configured)
            if self.email_user and self.email_password:
                # You would need to configure recipients
                recipients = os.getenv('EMAIL_RECIPIENTS', '').split(',')
                if recipients and recipients[0]:
                    results['email'] = self.send_email_notification(
                        subject, message, recipients, recommendations
                    )
            
            # Google Sheets export
            if self.google_credentials_file:
                sheet_url = self.export_to_google_sheets(recommendations)
                results['google_sheets'] = bool(sheet_url)
            
            # CSV export
            csv_file = self.export_to_csv(recommendations)
            results['csv'] = bool(csv_file)
            
            self.logger.info(f"Procurement alert sent. Results: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error sending procurement alert: {e}")
            return {'error': str(e)}
    
    def create_notification_summary(self, recommendations, market_analysis):
        """Create a summary for notifications."""
        try:
            if not recommendations:
                return "No recommendations found for the given criteria."
            
            summary = f"Found {len(recommendations)} supplier recommendations:\n\n"
            
            # Top 3 recommendations
            for i, rec in enumerate(recommendations[:3], 1):
                summary += f"{i}. {rec['supplier_name']} "
                summary += f"(${rec['total_cost']:.2f}, Score: {rec['supplier_score']:.2f}, Risk: {rec['risk_score']:.2f})\n"
            
            # Market insights
            if market_analysis:
                summary += f"\nMarket Insights:\n"
                summary += f"- Cost trend: {market_analysis.get('cost_trend', 'stable')}\n"
                summary += f"- Demand trend: {market_analysis.get('demand_trend', 'stable')}\n"
                summary += f"- Total market spend: ${market_analysis.get('total_spent', 0):,.2f}"
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error creating notification summary: {e}")
            return "Error generating summary"
