import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'ReRouteHer ESCO Matcher · TF-IDF',
  description: 'CPU-ready occupation coding from job title, skills and optional work length.',
  openGraph: {
    title: 'ReRouteHer ESCO Matcher · TF-IDF',
    description: 'Character n-gram occupation coding for varied job-title wording.',
    images: ['/og.png'],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'ReRouteHer ESCO Matcher · TF-IDF',
    description: 'Character n-gram occupation coding for varied job-title wording.',
    images: ['/og.png'],
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
